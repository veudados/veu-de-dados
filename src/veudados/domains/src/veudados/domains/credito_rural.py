import pandas as pd, numpy as np
from ..core import VeuDeDados

def caso_credito_rural(seed=11):
    """
    Crédito Rural: histórico de exclusão de clientes 'rural' face a 'urbana'.
    Objetivo: detetar viés de aprovação de crédito contra o meio rural.
    """
    np.random.seed(seed)
    n_rural, n_urb = 300, 700

    dados = pd.DataFrame({
        "zona": ["rural"]*n_rural + ["urbana"]*n_urb,
        "renda": np.r_[np.random.normal(800, 200, n_rural),
                       np.random.normal(2000, 500, n_urb)]
    })

    # Aprovações enviesadas: rural ~10%, urbana ~85.7% (exemplo forte p/ disparidade)
    aprov_rural = np.array([1]*30 + [0]*(n_rural-30))        # 30/300 = 10%
    aprov_urb   = np.array([1]*600 + [0]*(n_urb-600))        # 600/700 ≈ 85.7%
    dados["aprovado"] = np.r_[aprov_rural, aprov_urb]

    veu = VeuDeDados(
        dados=dados,
        alvo="aprovado",
        grupo_sensivel="zona",
        nome_contexto="Crédito Rural (simulado)"
    )
    return veu.aplicar_veu()
