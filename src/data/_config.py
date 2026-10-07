from pathlib import Path
import pandas as pd


BASE_PATH = Path(__file__).resolve().parent.parent.parent # starts in data and go src/ and then root
DATA_PATH = BASE_PATH / 'dados'
TRAIN_PATH = DATA_PATH / 'treino.csv'
VALIDATION_PATH = DATA_PATH / 'validacao.csv'
HOLDOUT_PATH = DATA_PATH / 'holdout_datas.csv'

# Hardcoded dates
TRAIN_START = pd.Timestamp("2011-01-29")
TRAIN_END = pd.Timestamp("2016-03-27")
VALIDATION_START = pd.Timestamp("2016-03-28")
VALIDATION_END = pd.Timestamp("2016-04-24")
HOLDOUT_START = pd.Timestamp("2016-04-25")
HOLDOUT_END = pd.Timestamp("2016-05-22")
DATE_COLUMN = "date"