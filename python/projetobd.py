"""
Classificação: um produto vai vender muito?

Melhorias em relação à versão original:
- Pipeline (pré-processamento + modelo) sem data leakage
- OneHotEncoder que aceita categorias novas
- StandardScaler nas variáveis numéricas
- Comparação: Regressão Logística x Árvore de Decisão
- Validação cruzada estratificada
- Métricas completas (precision, recall, F1, ROC-AUC)
- Threshold ótimo escolhido SEM olhar o conjunto de teste
- Importância das features e modelo salvo com joblib
"""

import logging
import os

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

RANDOM_STATE = 42
COL_NUMERICAS = ["preco", "estoque"]
COL_CATEGORICAS = ["categoria"]
ALVO = "vendeu_muito"


# ==========================================
# 1. CARREGAR DADOS
# ==========================================
def carregar_dados() -> pd.DataFrame:
    load_dotenv()
    url = os.getenv("MONGO_URL")
    banco = os.getenv("MONGO_DB")
    colecao = os.getenv("MONGO_COLLECTION")

    if not all([url, banco, colecao]):
        raise ValueError("Defina MONGO_URL, MONGO_DB e MONGO_COLLECTION no .env")

    with MongoClient(url) as client:
        # projeção: já traz só o que interessa, sem _id e __v
        docs = list(client[banco][colecao].find({}, {"_id": 0, "__v": 0}))

    log.info("%d documentos carregados", len(docs))
    return pd.DataFrame(docs)


# ==========================================
# 2. LIMPEZA
# ==========================================
def preparar_dados(df: pd.DataFrame):
    mapa = {"true": 1, "false": 0, "sim": 1, "nao": 0, "não": 0, "1": 1, "0": 0}
    df = df.copy()
    df[ALVO] = df[ALVO].astype(str).str.strip().str.lower().map(mapa)

    antes = len(df)
    df = df.dropna(subset=[ALVO] + COL_NUMERICAS + COL_CATEGORICAS)
    if len(df) < antes:
        log.warning("%d linhas removidas por valores ausentes/inválidos", antes - len(df))

    X = df[COL_NUMERICAS + COL_CATEGORICAS]
    y = df[ALVO].astype(int)

    log.info("Distribuição do alvo:\n%s", y.value_counts(normalize=True).round(2))
    return X, y


# ==========================================
# 3. PIPELINES
# ==========================================
def criar_preprocessador() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), COL_NUMERICAS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), COL_CATEGORICAS),
        ]
    )


def criar_modelos() -> dict:
    return {
        "Regressão Logística": Pipeline(
            [
                ("prep", criar_preprocessador()),
                (
                    "modelo",
                    LogisticRegression(
                        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "Árvore de Decisão": Pipeline(
            [
                ("prep", criar_preprocessador()),
                (
                    "modelo",
                    DecisionTreeClassifier(
                        max_depth=4,  # limita profundidade: evita overfitting
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


# ==========================================
# 4. THRESHOLD
# ==========================================
def escolher_threshold(modelo, X, y, cv) -> float:
    """Escolhe o threshold que maximiza o F1 usando probabilidades
    de validação cruzada (nunca o conjunto de teste)."""
    proba = cross_val_predict(modelo, X, y, cv=cv, method="predict_proba")[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return float(thresholds[np.argmax(f1[:-1])])


# ==========================================
# 5. IMPORTÂNCIA DAS FEATURES
# ==========================================
def mostrar_importancias(pipeline: Pipeline) -> None:
    nomes = pipeline.named_steps["prep"].get_feature_names_out()
    modelo = pipeline.named_steps["modelo"]

    if hasattr(modelo, "coef_"):
        valores = modelo.coef_[0]
        titulo = "Coeficientes (sinal indica direção do efeito)"
    else:
        valores = modelo.feature_importances_
        titulo = "Importância das features"

    ranking = pd.Series(valores, index=nomes).sort_values(key=abs, ascending=False)
    log.info("%s:\n%s", titulo, ranking.round(3))


# ==========================================
# 6. PREVISÃO PARA NOVOS PRODUTOS
# ==========================================
def prever_produto(pipeline, threshold, preco, estoque, categoria) -> tuple[str, float]:
    novo = pd.DataFrame(
        {"preco": [preco], "estoque": [estoque], "categoria": [categoria]}
    )
    prob = pipeline.predict_proba(novo)[0, 1]
    return ("Sim" if prob >= threshold else "Não"), prob


# ==========================================
# MAIN
# ==========================================
def main() -> None:
    X, y = preparar_dados(carregar_dados())

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    n_splits = int(min(5, y_treino.value_counts().min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    # --- compara modelos com validação cruzada (só no treino) ---
    resultados = {}
    for nome, pipe in criar_modelos().items():
        scores = cross_val_score(pipe, X_treino, y_treino, cv=cv, scoring="f1")
        resultados[nome] = scores.mean()
        log.info("%s | F1 (CV): %.3f ± %.3f", nome, scores.mean(), scores.std())

    melhor_nome = max(resultados, key=resultados.get)
    log.info("Melhor modelo: %s", melhor_nome)
    pipeline = criar_modelos()[melhor_nome]

    # --- threshold ótimo (sem olhar o teste) ---
    threshold = escolher_threshold(pipeline, X_treino, y_treino, cv)
    log.info("Threshold escolhido: %.2f (padrão seria 0.50)", threshold)

    # --- treino final e avaliação no teste ---
    pipeline.fit(X_treino, y_treino)
    proba_teste = pipeline.predict_proba(X_teste)[:, 1]
    y_pred = (proba_teste >= threshold).astype(int)

    print("\n========== RESULTADOS NO TESTE ==========")
    print(classification_report(y_teste, y_pred, target_names=["Não", "Sim"]))
    print("ROC-AUC:", round(roc_auc_score(y_teste, proba_teste), 3))
    print("\nMatriz de confusão (linhas = real, colunas = previsto):")
    print(confusion_matrix(y_teste, y_pred))

    mostrar_importancias(pipeline)

    # --- salva modelo + threshold ---
    joblib.dump({"pipeline": pipeline, "threshold": threshold}, "modelo_vendas.joblib")
    log.info("Modelo salvo em modelo_vendas.joblib")

    # --- exemplo de uso ---
    resultado, prob = prever_produto(pipeline, threshold, 250, 20, "Informática")
    print(f"\nVendeu muito? {resultado} (probabilidade: {prob:.1%})")


if __name__ == "__main__":
    main()