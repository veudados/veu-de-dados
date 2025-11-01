import pandas as pd, numpy as np
from ..core import VeuDeDados

def caso_amazon_2018(seed=42):
    np.random.seed(seed)
    dados = pd.DataFrame({
        "genero": ["homem"]*800 + ["mulher"]*200,
        "palavras": (["tecnologia","engenharia"]*600 +
                     ["mulheres","clube feminino"]*200 +
                     ["geral"]*200),
        "contratado_hist": [1]*640 + [0]*160 + [1]*40 + [0]*160
    })
    dados["score_ia"] = dados["contratado_hist"]
    mask = dados["palavras"].str.contains("mulher|mulheres|femin", case=False, na=False)
    dados.loc[mask, "score_ia"] = 0

    veu = VeuDeDados(dados, alvo="score_ia", grupo_sensivel="genero",
                     nome_contexto="Amazon Recrutamento 2018 (simulado)")
    return veu.aplicar_veu()
