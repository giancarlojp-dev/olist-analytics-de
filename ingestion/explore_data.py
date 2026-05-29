"""
ingestion/explore_data.py
──────────────────────────────────────────────────────────────────────────────
Exploración inicial de los 9 CSVs de Olist.
Ejecutar ANTES de load_to_bigquery.py para entender la estructura,
detectar nulos, tipos de datos y relaciones entre tablas.

Uso:
    python ingestion/explore_data.py
    python ingestion/explore_data.py --table orders
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

RAW_DATA_PATH = Path(os.getenv("RAW_DATA_PATH", "./data/raw"))

# ── Mapa de archivos ──────────────────────────────────────────────────────────
CSV_FILES: dict[str, str] = {
    "orders":               "olist_orders_dataset.csv",
    "order_items":          "olist_order_items_dataset.csv",
    "customers":            "olist_customers_dataset.csv",
    "products":             "olist_products_dataset.csv",
    "sellers":              "olist_sellers_dataset.csv",
    "payments":             "olist_order_payments_dataset.csv",
    "reviews":              "olist_order_reviews_dataset.csv",
    "geolocation":          "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

SEP = "─" * 60


def explore_table(name: str, csv_file: str) -> None:
    """Imprime un resumen completo de una tabla CSV."""
    path = RAW_DATA_PATH / csv_file

    if not path.exists():
        print(f"\n[SKIP] {name}: archivo no encontrado en {path}")
        return

    df = pd.read_csv(path, low_memory=False)

    print(f"\n{SEP}")
    print(f"  TABLA: {name.upper()}  ({csv_file})")
    print(SEP)

    # Dimensiones
    print(f"  Filas      : {len(df):>10,}")
    print(f"  Columnas   : {len(df.columns):>10}")

    # Columnas y tipos
    print(f"\n  {'Columna':<40} {'Dtype':<15} {'Nulos':>8} {'Nulos %':>8}")
    print(f"  {'─'*40} {'─'*15} {'─'*8} {'─'*8}")
    for col in df.columns:
        nulls = df[col].isna().sum()
        pct   = nulls / len(df) * 100
        flag  = " ⚠" if pct > 20 else ""
        print(f"  {col:<40} {str(df[col].dtype):<15} {nulls:>8,} {pct:>7.1f}%{flag}")

    # Muestra de valores únicos para columnas categóricas
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        print(f"\n  Valores únicos en columnas de texto:")
        for col in cat_cols[:6]:  # máximo 6 para no saturar la pantalla
            n_unique = df[col].nunique()
            sample   = df[col].dropna().unique()[:5].tolist()
            print(f"    {col:<35} {n_unique:>6} únicos  →  {sample}")

    # Estadísticas numéricas rápidas
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        print(f"\n  Columnas numéricas (min / max / mean):")
        for col in num_cols[:6]:
            s = df[col].dropna()
            print(
                f"    {col:<35}  min={s.min():.2f}  max={s.max():.2f}  "
                f"mean={s.mean():.2f}"
            )

    print()


def check_relationships() -> None:
    """Verifica integridad referencial entre las tablas principales."""
    print(f"\n{SEP}")
    print("  VERIFICACIÓN DE RELACIONES (integridad referencial)")
    print(SEP)

    files = {
        k: RAW_DATA_PATH / v
        for k, v in CSV_FILES.items()
        if (RAW_DATA_PATH / v).exists()
    }

    if "orders" not in files or "order_items" not in files:
        print("  Faltan tablas orders u order_items — omitiendo check.")
        return

    orders      = pd.read_csv(files["orders"],      usecols=["order_id"])
    order_items = pd.read_csv(files["order_items"],  usecols=["order_id", "product_id", "seller_id"])

    # ¿Todos los order_ids de order_items existen en orders?
    orphan_items = order_items[~order_items["order_id"].isin(orders["order_id"])]
    print(f"  order_items con order_id huérfano  : {len(orphan_items):,}")

    if "customers" in files:
        customers = pd.read_csv(files["customers"], usecols=["customer_id"])
        orders_full = pd.read_csv(files["orders"], usecols=["order_id", "customer_id"])
        orphan_orders = orders_full[~orders_full["customer_id"].isin(customers["customer_id"])]
        print(f"  orders con customer_id huérfano    : {len(orphan_orders):,}")

    if "products" in files:
        products = pd.read_csv(files["products"], usecols=["product_id"])
        orphan_products = order_items[~order_items["product_id"].isin(products["product_id"])]
        print(f"  order_items con product_id huérfano: {len(orphan_products):,}")

    if "sellers" in files:
        sellers = pd.read_csv(files["sellers"], usecols=["seller_id"])
        orphan_sellers = order_items[~order_items["seller_id"].isin(sellers["seller_id"])]
        print(f"  order_items con seller_id huérfano : {len(orphan_sellers):,}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exploración inicial de los CSVs de Olist"
    )
    parser.add_argument(
        "--table",
        choices=list(CSV_FILES.keys()),
        help="Explorar solo una tabla específica",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  OLIST DATA EXPLORATION")
    print(f"  Directorio: {RAW_DATA_PATH.resolve()}")
    print(f"{'='*60}")

    # Verificar que el directorio existe
    if not RAW_DATA_PATH.exists():
        print(f"\n[ERROR] El directorio {RAW_DATA_PATH} no existe.")
        print("  Crea data/raw/ y descarga los CSVs de Kaggle primero.")
        return

    # Listar archivos presentes
    csv_found = list(RAW_DATA_PATH.glob("*.csv"))
    print(f"\n  Archivos CSV encontrados en data/raw/: {len(csv_found)}")
    for f in sorted(csv_found):
        size_mb = f.stat().st_size / 1_048_576
        print(f"    {f.name:<50} {size_mb:>6.1f} MB")

    # Explorar tablas
    tables_to_explore = (
        {args.table: CSV_FILES[args.table]} if args.table else CSV_FILES
    )

    for name, csv_file in tables_to_explore.items():
        explore_table(name, csv_file)

    # Verificar relaciones solo en exploración completa
    if not args.table:
        check_relationships()

    print(f"{'='*60}")
    print("  Exploración completa. Siguiente paso: load_to_bigquery.py --dry-run")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()