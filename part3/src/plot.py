import matplotlib.pyplot as plt
import os
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

DIR = "../outputs/"
FOLDER = os.listdir(DIR)
MODELOS = ["arvore", "boosting"]


def plotSeries():
    for file in FOLDER:
        if not file.startswith('previsoes'):
            continue

        # Abre o .csv e filtra o modelo
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        y = pd.read_csv(DIR + file, parse_dates=["date"]).sort_values("date")

        for tipo in MODELOS:
            z = y[y['modelo'] == tipo]

            # Plota a série
            axes.plot(z.date, z.yhat, label=tipo, alpha=0.9)
            axes.set_title(f"{file}")

        # Salva a imagem
        axes.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(f"../figures/{file[:-4]}.png", dpi=120)


if __name__ == "__main__":
    plotSeries()

