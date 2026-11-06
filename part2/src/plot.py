import matplotlib.pyplot as plt
import os
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

DIR = "../outputs/"
FOLDER = os.listdir(DIR)

def plotSeries():
    for file in FOLDER:
        if not file.startswith('previsoes'):
            continue

        # Abre o .csv
        y = pd.read_csv(DIR + file, parse_dates=["date"]).sort_values("date")

        # Plota a série
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes.plot(y.date, y.yhat, c="#1B365D", alpha=0.9)
        axes.set_title(f"{file}")

        # Salva a imagem
        fig.tight_layout()
        fig.savefig(f"../figures/{file[:-4]}.png", dpi=120)


if __name__ == "__main__":
    plotSeries()