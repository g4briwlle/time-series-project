import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from data.reader import DataReader, SERIES

def plotSeries(data):
    for nome in SERIES:
        y = data.get_series(nome)

        # Plota a série
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        axes[0].plot(y.index, y.values, c="#1B365D", alpha=0.9)
        axes[0].set_title(f"{nome}")

        # Plota o ACF e PACF
        plot_acf(y.dropna(), lags=40, ax=axes[1], title='ACF', color="#4A777A")
        plot_pacf(y.dropna(), lags=40, ax=axes[2], method="ywm", title='PACF', color="#4A777A")

        # Muda a cor do intervalo de confiança
        for collection in axes[1].collections:
            if isinstance(collection, plt.matplotlib.collections.PolyCollection):
                collection.set_color("#D9E1E8")
                collection.set_alpha(0.8)

        for collection in axes[2].collections:
                if isinstance(collection, plt.matplotlib.collections.PolyCollection):
                    collection.set_color("#D9E1E8")
                    collection.set_alpha(0.8)

        # Salva a imagem
        fig.tight_layout()
        fig.savefig(f"./figures/{nome}.png", dpi=120)


if __name__ == "__main__":
    train = DataReader("train")
    plotSeries(train)
