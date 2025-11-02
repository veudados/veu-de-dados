phyton
import pandas as pd, numpy as np
from ..core import VeuDeDados

def caso_iefp_emprego(seed=7):
    """
    IEFP / Emprego: exclusão territorial (litoral vs interior).
    Objetivo: detetar viés contra candidatos do 'interior'.
    """
    np.random.seed(seed)
    n_interior, n_litoral = 300, 700

    dados = pd.DataFrame({
        "territorio": ["interior"]*n_interior + ["litoral"]*n_litoral,
        "qualificacao": np.random.choice([0,1,2,3,4], n_interior+n_litoral, p=[0.1,0.2,0.3,0.3,0.1]),
        # taxa de contratação enviesada: interior 15%, litoral 70%
        "contratado":    [1]*45 + [0]*(n_interior-45) + [1]*490 + [0]*(n_litoral-490)
    })

    veu = VeuDeDados(
        dados=dados,
        alvo="contratado",
        grupo_sensivel="territorio",
        nome_contexto="IEFP Emprego (simulado)"
    )
    return veu.aplicar_veu()
