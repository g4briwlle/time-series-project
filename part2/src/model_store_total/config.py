import pandas as pd

from pathlib import Path


BASE_PATH = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_PATH / 'dados'
OUTPUT_DIR = BASE_PATH / 'outputs'
SERIE = "store_total"
PRECO_COL = f"preco_{SERIE}"
ORIGEM = pd.Timestamp("2016-03-27")
VAL_START = pd.Timestamp("2016-03-28")
VAL_END = pd.Timestamp("2016-04-24")
LAG = 28

VAL_DATES = pd.date_range(VAL_START, VAL_END, freq="D")