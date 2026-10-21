import logging
import math
import os
import threading

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from projeto_vendas import (
    calcular_importancias,
    carregar_dados,
    preparar_dados,
    treinar_e_avaliar,
)

log = logging.getLogger(__name__)

app = Flask(__name__)

MODELO_PATH = "modelo_vendas.joblib"

# Guarda o último modelo treinado. Trocamos o dicionário inteiro de uma vez,
# assim uma requisição nunca enxerga um estado pela metade.
ESTADO = {"atual": None}
_trava = threading.Lock()

NOMES_BONITOS = {"preco": "Preço", "estoque": "Estoque"}


def nome_amigavel(nome_feature: str) -> str:
    """'cat__categoria_Informática' -> 'Categoria: Informática'."""
    nome = nome_feature.split("__", 1)[-1]
    if nome.startswith("categoria_"):
        return "Categoria: " + nome[len("categoria_"):]
    return NOMES_BONITOS.get(nome, nome)


def treinar():
    with _trava:
        X, y = preparar_dados(carregar_dados())
        r = treinar_e_avaliar(X, y)

        importancias = calcular_importancias(r["pipeline"])
        tipo = "coeficiente" if hasattr(r["pipeline"].named_steps["modelo"], "coef_") else "importância"

        ESTADO["atual"] = {
            "pipeline": r["pipeline"],
            "threshold": r["threshold"],
            "modelo_nome": r["modelo_nome"],
            "categorias": r["categorias"],
            "metricas": r["metricas"],
            "matriz": r["matriz"],
            "comparacao": r["comparacao"],
            "importancias": [
                {"nome": nome_amigavel(n), "valor": float(v)}
                for n, v in importancias.head(8).items()
            ],
            "tipo_importancia": tipo,
            "total": r["total"],
            "teste": r["teste"],
        }

        joblib.dump({"pipeline": r["pipeline"], "threshold": r["threshold"]}, MODELO_PATH)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/modelo")
def info_modelo():
    e = ESTADO["atual"]
    if e is None:
        return jsonify(erro="O modelo ainda não foi treinado. Clique em “Retreinar”."), 503

    return jsonify(
        modelo=e["modelo_nome"],
        limiar=e["threshold"],
        categorias=e["categorias"],
        metricas=e["metricas"],
        matriz=e["matriz"],
        comparacao=e["comparacao"],
        importancias=e["importancias"],
        tipo_importancia=e["tipo_importancia"],
        total=e["total"],
        teste=e["teste"],
    )


@app.route("/api/treinar", methods=["POST"])
def retreinar():
    try:
        treinar()
    except Exception as erro:
        log.exception("Falha ao treinar")
        return jsonify(erro=f"Não foi possível treinar: {erro}"), 500
    return info_modelo()


@app.route("/api/prever", methods=["POST"])
def prever():
    e = ESTADO["atual"]
    if e is None:
        return jsonify(erro="O modelo ainda não foi treinado."), 503

    dados = request.get_json(silent=True) or {}

    try:
        preco = float(dados["preco"])
        estoque = float(dados["estoque"])
        categoria = str(dados["categoria"])
    except (KeyError, ValueError, TypeError):
        return jsonify(erro="Preencha preço, estoque e categoria com valores válidos."), 400

    if not (math.isfinite(preco) and math.isfinite(estoque)) or preco < 0 or estoque < 0:
        return jsonify(erro="Preço e estoque devem ser números maiores ou iguais a zero."), 400

    if categoria not in e["categorias"]:
        return jsonify(erro="Categoria desconhecida."), 400

    # O pipeline recebe os dados "crus": ele mesmo escala os números
    # e transforma a categoria em colunas 0/1.
    novo_produto = pd.DataFrame(
        {"preco": [preco], "estoque": [estoque], "categoria": [categoria]}
    )

    pipeline = e["pipeline"]
    posicao_sim = list(pipeline.classes_).index(1)
    prob_sim = float(pipeline.predict_proba(novo_produto)[0][posicao_sim])

    # Usa o threshold escolhido no treino (e não o 0,5 padrão do predict)
    vendeu_muito = prob_sim >= e["threshold"]

    return jsonify(
        vendeu_muito=vendeu_muito,
        resultado="Sim" if vendeu_muito else "Não",
        probabilidade=prob_sim,
        limiar=e["threshold"],
    )


if __name__ == "__main__":
    try:
        treinar()
    except Exception:
        # O servidor sobe mesmo assim: a página mostra o erro e dá para
        # tentar de novo pelo botão "Retreinar" depois de corrigir o .env
        log.exception("Não foi possível treinar na inicialização")

    app.run(debug=os.getenv("FLASK_DEBUG") == "1", use_reloader=False)