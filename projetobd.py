import os

from dotenv import load_dotenv

from pymongo import MongoClient

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix


# Carrega o .env
load_dotenv()


# Pega as informações do .env
MONGO_URL = os.getenv("MONGO_URL")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")


# Conecta ao MongoDB Atlas
client = MongoClient(MONGO_URL)


# Seleciona o banco
db = client[MONGO_DB]


# Seleciona a collection
colecao = db[MONGO_COLLECTION]


# Busca os documentos
dados = list(colecao.find())


# Transforma em DataFrame
df = pd.DataFrame(dados)


# ==========================================
# TRANSFORMAR CATEGORIA EM NÚMEROS
# ==========================================

df = pd.get_dummies(
    df,
    columns=["categoria"]
)


# ==========================================
# TRANSFORMAR "SIM" E "NAO" EM 1 E 0
# ==========================================

df["vendeu_muito"] = df["vendeu_muito"].map({
    "sim": 1,
    "nao": 0
})


print("\nDados:")
print(df)


# ==========================================
# X E Y
# ==========================================

X = df.drop(
    columns=["vendeu_muito", "nome", "_id", "__v"],
    errors="ignore"
)

y = df["vendeu_muito"]


print("\nX:")
print(X)

print("\nY:")
print(y)


# ==========================================
# TREINO E TESTE
# ==========================================

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# CRIAR MODELO
# ==========================================

modelo = LogisticRegression()


# ==========================================
# TREINAR MODELO
# ==========================================

modelo.fit(
    X_treino,
    y_treino
)


# ==========================================
# FAZER PREVISÕES
# ==========================================

y_pred = modelo.predict(X_teste)


# ==========================================
# RESULTADOS
# ==========================================

print("\n========== RESULTADOS ==========")

accuracy = accuracy_score(
    y_teste,
    y_pred
)

print("\nResultado das previsões:")

for real, previsto in zip(y_teste, y_pred):
    print(
        f"Real: {real} | Previsto: {previsto}"
    )

print("Accuracy:", accuracy)


print("\nMatriz de confusão:")

print(
    confusion_matrix(
        y_teste,
        y_pred
    )
)

#================================
#     Novo Produto
#================================

novo_produto = pd.DataFrame({
    "preco": [250],
    "estoque": [20],
    "categoria_Informática": [True]
})

previsao = modelo.predict(novo_produto)

resultado = "Sim" if previsao[0] == 1 else "Não"

print(f"Vendeu muito? {resultado}")