import numpy as np
import torch
from torch import nn

from prepare_data import HORIZONTE, carregar_dados, preparar_corte

HISTORICOS = (28, 56)


def janelas_treino(y, covariaveis, historico):
    if len(y) != len(covariaveis):
        raise ValueError("Vendas e covariáveis devem cobrir os mesmos dias de treino.")
    if historico < 1 or len(y) < historico + HORIZONTE:
        raise ValueError("O treino deve cobrir o histórico e os 28 dias alvo.")


    primeiro_alvo = np.arange(historico, len(y) - HORIZONTE + 1)[:, None]
    indices_passado = primeiro_alvo + np.arange(-historico, 0)
    indices_alvo = primeiro_alvo + np.arange(HORIZONTE)

    vendas_passadas = torch.as_tensor(y[indices_passado], dtype=torch.float32)
    covariaveis_alvo = torch.as_tensor(covariaveis[indices_alvo], dtype=torch.float32)
    vendas_alvo = torch.as_tensor(y[indices_alvo], dtype=torch.float32)
    return vendas_passadas.unsqueeze(-1), covariaveis_alvo, vendas_alvo


class RedeFoods(nn.Module):
    def __init__(self, n_covariaveis):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.saida = nn.Sequential(
            nn.Linear(16 + n_covariaveis, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, historico, covariaveis):
        _, (estado_oculto, _) = self.lstm(historico)
        # Repete o resumo do histórico uma vez para cada dia a prever.
        resumo = estado_oculto[-1].unsqueeze(1).expand(-1, covariaveis.shape[1], -1)
        entradas = torch.cat((resumo, covariaveis), dim=-1)
        return self.saida(entradas).squeeze(-1)


vendas, entradas, datas_validacao = carregar_dados()
y, cov_treino, _, _, _ = preparar_corte(vendas, entradas, datas_validacao)
modelo = RedeFoods(cov_treino.shape[1]).eval()
for historico in HISTORICOS:
    passado, covariaveis, alvos = janelas_treino(y, cov_treino, historico)
    with torch.inference_mode():
        saida = modelo(passado[:1], covariaveis[:1])
    print(
        f"Histórico de {historico} dias | {len(alvos)} janelas | "
        f"vendas passadas {tuple(passado.shape)} | "
        f"covariáveis futuras {tuple(covariaveis.shape)} | "
        f"alvos {tuple(alvos.shape)} | saída para uma janela {tuple(saida.shape)}"
    )