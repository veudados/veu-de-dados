import pandas as pd, numpy as np
from ..core import VeuDeDados

def caso_saude_triagem(seed=33):
    """
    Saúde Pública / Triagem Hospitalar: viés de género e território.
    Objetivo: detetar subavaliação do risco em mulheres e residentes do interior.
    """
    np.random.seed(seed)
    n = 1000
    dados = pd.DataFrame({
        "genero": np.random.choice(["homem","mulher"], n, p=[0.45, 0.55]),
        "zona": np.random.choice(["urbana","interior"], n, p=[0.7, 0.3]),
        "idade": np.random.normal(50, 18, n).astype(int)
    })

    # Score real de risco (com base em idade)
    dados["risco_real"] = np.clip((dados["idade"]/100) + np.random.rand(n)*0.5, 0, 1)

    # IA enviesada: reduz ligeiramente o risco de mulheres e pessoas do interior
    dados["score_ia"] = dados["risco_real"]
    dados.loc[dados["genero"]=="mulher", "score_ia"] *= 0.85
    dados.loc[dados["zona"]=="interior", "score_ia"] *= 0.9

    veu = VeuDeDados(
        dados=dados,
        alvo="score_ia",
        grupo_sensivel="genero",
        nome_contexto="Triagem Hospitalar (viés de género e território)"
    )
    return veu.aplicar_veu()
