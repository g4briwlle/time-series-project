import pandas as pd

from pathlib import Path


BASE_PATH = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_PATH / 'dados'
OUTPUT_DIR = BASE_PATH / 'outputs'
OUTPUT_DIR_TREE = BASE_PATH.parent / 'part3' / 'outputs'
SERIE = "store_total"
PRECO_COL = f"preco_{SERIE}"
ORIGEM = pd.Timestamp("2016-03-27")
VAL_START = pd.Timestamp("2016-03-28")
VAL_END = pd.Timestamp("2016-04-24")
LAG = 28
LAGS_RECURSIVE = [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]

VAL_DATES = pd.date_range(VAL_START, VAL_END, freq="D")