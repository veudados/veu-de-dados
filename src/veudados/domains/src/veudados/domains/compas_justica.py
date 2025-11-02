import pandas as pd, numpy as np
from ..core import VeuDeDados

def caso_compas_justica(seed=11):
    """
    COMPAS / Justiça Criminal: viés racial (negros vs brancos).
    Objetivo: demonstrar como o algoritmo de previsão de reincidência
    penal apresentava taxas de falsos positivos maiores para negros.
    """
    np.random.seed(seed)
    n_negros, n_brancos = 500, 500

    # Dados simulados baseados em padrões observados no caso real ProPublica (2016)
    dados = pd.DataFrame({
        "raca": ["negro"]*n_negros + ["branco"]*n_brancos,
        "idade": np.random.randint(18, 60, n_negros+n_brancos),
        "antecedentes": np.random.poisson(1.5, n_negros+n_brancos),
        "reincidente_real": np.random.binomial(1, 0.4, n_negros+n_brancos)
    })

    # Simulação do viés: algoritmo tende a marcar mais negros como "alto risco"
    dados["score_ia"] = dados["reincidente_real"]
    mask = (dados["raca"] == "negro") & (np.random.rand(n_negros+n_brancos) < 0.25)
    dados.loc[mask, "score_ia"] = 1  # viés: aumenta falsos positivos para negros

    veu = VeuDeDados(
        dados=dados,
        alvo="score_ia",
        grupo_sensivel="raca",
        nome_contexto="COMPAS Justiça Criminal (simulado)"
    )
    return veu.aplicar_veu()
