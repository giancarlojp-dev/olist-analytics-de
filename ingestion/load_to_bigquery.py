"""
ingestion/load_to_bigquery.py
──────────────────────────────────────────────────────────────────────────────
Lee los 9 CSVs de Olist, aplica validaciones básicas y los carga al
dataset `raw` de BigQuery usando WRITE_TRUNCATE (carga completa).

Uso:
    python ingestion/load_to_bigquery.py --dry-run   # valida sin subir
    python ingestion/load_to_bigquery.py             # carga todo
    python ingestion/load_to_bigquery.py --table orders
    python ingestion/load_to_bigquery.py --table orders customers payments
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError
from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Variables de entorno ──────────────────────────────────────────────────────
load_dotenv()

GCP_PROJECT_ID  = os.getenv("GCP_PROJECT_ID")
BQ_RAW_DATASET  = os.getenv("BQ_RAW_DATASET", "raw")
BQ_LOCATION     = os.getenv("BQ_LOCATION", "US")
RAW_DATA_PATH   = Path(os.getenv("RAW_DATA_PATH", "./data/raw"))


# ── Especificaciones de cada tabla ────────────────────────────────────────────
@dataclass
class TableSpec:
    bq_table:         str
    csv_file:         str
    required_columns: list
    parse_dates:      list  = field(default_factory=list)
    dtype_overrides:  dict  = field(default_factory=dict)
    description:      str   = ""


TABLE_REGISTRY: dict[str, TableSpec] = {

    "orders": TableSpec(
        bq_table="orders",
        csv_file="olist_orders_dataset.csv",
        required_columns=["order_id", "customer_id", "order_status"],
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
        description="Tabla central: un registro por pedido",
    ),

    "order_items": TableSpec(
        bq_table="order_items",
        csv_file="olist_order_items_dataset.csv",
        required_columns=["order_id", "product_id", "seller_id"],
        parse_dates=["shipping_limit_date"],
        dtype_overrides={"price": float, "freight_value": float},
        description="Lineas de pedido: un registro por item dentro del pedido",
    ),

    "customers": TableSpec(
        bq_table="customers",
        csv_file="olist_customers_dataset.csv",
        required_columns=["customer_id", "customer_unique_id", "customer_state"],
        dtype_overrides={"customer_zip_code_prefix": str},
        description="Dimension cliente",
    ),

    "products": TableSpec(
        bq_table="products",
        csv_file="olist_products_dataset.csv",
        required_columns=["product_id"],
        dtype_overrides={
            "product_weight_g":   float,
            "product_length_cm":  float,
            "product_height_cm":  float,
            "product_width_cm":   float,
        },
        description="Catalogo de productos con dimensiones fisicas",
    ),

    "sellers": TableSpec(
        bq_table="sellers",
        csv_file="olist_sellers_dataset.csv",
        required_columns=["seller_id", "seller_state"],
        dtype_overrides={"seller_zip_code_prefix": str},
        description="Dimension vendedores",
    ),

    "payments": TableSpec(
        bq_table="payments",
        csv_file="olist_order_payments_dataset.csv",
        required_columns=["order_id", "payment_type", "payment_value"],
        dtype_overrides={"payment_value": float},
        description="Pagos por pedido (puede haber varios por order_id)",
    ),

    "reviews": TableSpec(
        bq_table="reviews",
        csv_file="olist_order_reviews_dataset.csv",
        required_columns=["review_id", "order_id", "review_score"],
        parse_dates=["review_creation_date", "review_answer_timestamp"],
        dtype_overrides={"review_score": "Int64"},
        description="Reviews de clientes ligadas a pedidos",
    ),

    "geolocation": TableSpec(
        bq_table="geolocation",
        csv_file="olist_geolocation_dataset.csv",
        required_columns=["geolocation_zip_code_prefix", "geolocation_state"],
        dtype_overrides={
            "geolocation_zip_code_prefix": str,
            "geolocation_lat":             float,
            "geolocation_lng":             float,
        },
        description="Coordenadas lat/lng por codigo postal de Brasil",
    ),

    "category_translation": TableSpec(
        bq_table="category_translation",
        csv_file="product_category_name_translation.csv",
        required_columns=[
            "product_category_name",
            "product_category_name_english",
        ],
        description="Traduccion de categorias: Portugues a Ingles",
    ),
}


# ── Validaciones ──────────────────────────────────────────────────────────────

def validate_dataframe(df: pd.DataFrame, spec: TableSpec) -> list:
    errors = []

    if df.empty:
        errors.append("El DataFrame esta vacio — 0 filas cargadas")
        return errors

    for col in spec.required_columns:
        if col not in df.columns:
            errors.append(f"Columna requerida '{col}' NO existe en el CSV")
        elif df[col].isna().all():
            errors.append(f"Columna requerida '{col}' esta completamente vacia")

    pk_map = {
        "orders":    "order_id",
        "customers": "customer_id",
        "products":  "product_id",
        "sellers":   "seller_id",
        "reviews":   "review_id",
    }
    if spec.bq_table in pk_map:
        pk_col = pk_map[spec.bq_table]
        if pk_col in df.columns:
            n_dups = int(df[pk_col].duplicated().sum())
            if n_dups > 0:
                errors.append(
                    f"Clave primaria '{pk_col}' tiene {n_dups:,} duplicados"
                )

    return errors


def print_null_summary(df: pd.DataFrame, table_name: str) -> None:
    null_pct = (df.isna().sum() / len(df) * 100).round(1)
    high_null = null_pct[null_pct > 20]
    if not high_null.empty:
        log.warning("  Columnas con >20%% nulos en [%s]:", table_name)
        for col, pct in high_null.items():
            log.warning("    %-40s %.1f%%", col, pct)


# ── BigQuery helpers ──────────────────────────────────────────────────────────

def get_bq_client() -> bigquery.Client:
    if not GCP_PROJECT_ID:
        raise EnvironmentError(
            "\n[ERROR] GCP_PROJECT_ID no definido.\n"
            "  Completa el archivo .env con tu project ID de Google Cloud.\n"
        )
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not key_path or not Path(key_path).exists():
        raise EnvironmentError(
            f"\n[ERROR] GOOGLE_APPLICATION_CREDENTIALS invalido: {key_path}\n"
            "  Descarga el JSON key de tu Service Account y actualiza .env\n"
        )
    log.info("Autenticando con service account: %s", key_path)
    return bigquery.Client(project=GCP_PROJECT_ID)


def ensure_dataset_exists(client: bigquery.Client, dataset_id: str) -> None:
    full_id = f"{GCP_PROJECT_ID}.{dataset_id}"
    try:
        client.get_dataset(full_id)
        log.info("Dataset ya existe    : %s", full_id)
    except Exception:
        dataset = bigquery.Dataset(full_id)
        dataset.location = BQ_LOCATION
        client.create_dataset(dataset, exists_ok=True)
        log.info("Dataset creado       : %s  (location=%s)", full_id, BQ_LOCATION)


def upload_to_bigquery(
    client: bigquery.Client,
    df:     pd.DataFrame,
    spec:   TableSpec,
) -> None:
    destination = f"{GCP_PROJECT_ID}.{BQ_RAW_DATASET}.{spec.bq_table}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True,
        source_format=bigquery.SourceFormat.PARQUET,
    )

    log.info(
        "Subiendo  %-25s (%s filas)  ->  %s",
        spec.bq_table,
        f"{len(df):,}",
        destination,
    )
    t0  = time.time()
    job = client.load_table_from_dataframe(df, destination, job_config=job_config)
    job.result()

    log.info("  OK  %-25s cargada en %.1fs", spec.bq_table, time.time() - t0)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def process_single_table(
    spec:    TableSpec,
    client:  Optional[bigquery.Client],
    dry_run: bool = False,
) -> tuple:
    csv_path = RAW_DATA_PATH / spec.csv_file

    if not csv_path.exists():
        msg = f"CSV no encontrado: {csv_path}"
        log.error(msg)
        return False, msg

    log.info("Leyendo  %s ...", csv_path.name)
    try:
        df = pd.read_csv(
            csv_path,
            dtype       = spec.dtype_overrides if spec.dtype_overrides else None,
            parse_dates = spec.parse_dates     if spec.parse_dates     else False,
            low_memory  = False,
        )
    except Exception as exc:
        msg = f"Error leyendo {csv_path.name}: {exc}"
        log.error(msg)
        return False, msg

    log.info("  Dimensiones: %d filas x %d columnas", len(df), len(df.columns))

    errors = validate_dataframe(df, spec)
    for err in errors:
        log.warning("  [VALIDACION] %s", err)

    print_null_summary(df, spec.bq_table)

    if dry_run:
        log.info("  [DRY-RUN] OK — no se sube nada para %s", spec.bq_table)
        return True, "dry-run ok"

    try:
        upload_to_bigquery(client, df, spec)
    except (GoogleCloudError, Exception) as exc:
        msg = f"Error subiendo {spec.bq_table}: {exc}"
        log.error(msg)
        return False, msg

    return True, "cargada"


def run_pipeline(
    tables_filter: Optional[list] = None,
    dry_run:       bool           = False,
) -> None:

    log.info("=" * 60)
    log.info("  OLIST INGESTION PIPELINE")
    log.info("  Proyecto   : %s", GCP_PROJECT_ID or "NO DEFINIDO")
    log.info("  Dataset BQ : %s", BQ_RAW_DATASET)
    log.info("  Location   : %s", BQ_LOCATION)
    log.info("  Data dir   : %s", RAW_DATA_PATH.resolve())
    log.info("  Dry-run    : %s", dry_run)
    log.info("=" * 60)

    if tables_filter:
        invalid = set(tables_filter) - set(TABLE_REGISTRY)
        if invalid:
            log.error(
                "Tablas invalidas: %s\nValidas: %s",
                sorted(invalid),
                sorted(TABLE_REGISTRY.keys()),
            )
            sys.exit(1)
        targets = {k: v for k, v in TABLE_REGISTRY.items() if k in tables_filter}
    else:
        targets = TABLE_REGISTRY

    log.info("Tablas a procesar: %s", list(targets.keys()))

    client: Optional[bigquery.Client] = None
    if not dry_run:
        client = get_bq_client()
        ensure_dataset_exists(client, BQ_RAW_DATASET)

    results = {}
    for name, spec in tqdm(targets.items(), desc="Cargando tablas", unit="tabla"):
        log.info("─" * 40)
        ok, msg       = process_single_table(spec, client, dry_run=dry_run)
        results[name] = (ok, msg)

    log.info("=" * 60)
    passed = [t for t, (ok, _) in results.items() if ok]
    failed = [t for t, (ok, _) in results.items() if not ok]

    log.info("  RESULTADO FINAL")
    log.info("  Exitosas : %d -> %s", len(passed), passed)

    if failed:
        log.error("  Fallidas : %d -> %s", len(failed), failed)
        for t in failed:
            log.error("    [%s] %s", t, results[t][1])
        sys.exit(1)

    if dry_run:
        log.info("  [DRY-RUN] Validacion completada. Ejecuta sin --dry-run para cargar.")
    else:
        log.info("  Todas las tablas cargadas al dataset '%s'", BQ_RAW_DATASET)
        log.info(
            "  Verifica en: https://console.cloud.google.com/bigquery?project=%s",
            GCP_PROJECT_ID,
        )
    log.info("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Carga los 9 CSVs de Olist al dataset raw de BigQuery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python ingestion/load_to_bigquery.py --dry-run
  python ingestion/load_to_bigquery.py
  python ingestion/load_to_bigquery.py --table orders
  python ingestion/load_to_bigquery.py --table orders customers payments
        """,
    )
    parser.add_argument(
        "--table",
        nargs="*",
        metavar="TABLA",
        help=f"Procesar solo estas tablas. Opciones: {sorted(TABLE_REGISTRY)}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida los CSVs localmente sin subir nada a BigQuery",
    )
    args = parser.parse_args()
    run_pipeline(tables_filter=args.table, dry_run=args.dry_run)


if __name__ == "__main__":
    main()