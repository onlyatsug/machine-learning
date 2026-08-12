"""
PASSO 0 — Pré-processamento compartilhado
Adult Census Income Dataset (UCI) — usando os arquivos originais
adult.data + adult.test (do zip baixado em archive.ics.uci.edu)

Rodem este script UMA VEZ, juntos, no início.
Coloquem adult.data e adult.test na mesma pasta do script.

Gera dois arquivos:
    - adult_preprocessado.csv   (dados prontos, já codificados e normalizados)
    - train_test_split.npz      (X_train, X_test, y_train, y_test já separados)

A partir daqui:
    - Pessoa A (Árvore de Decisão) carrega o .npz e treina o modelo
    - Pessoa B (K-Means) carrega o .npz (ou o .csv) e faz o agrupamento

Requisitos: pip install scikit-learn pandas numpy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ---------------------------------------------------------
# 1. CARREGAR OS DADOS (arquivos locais, sem necessidade de internet)
# ---------------------------------------------------------
COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

# adult.data: dados de treino originais, sem cabeçalho
df_train = pd.read_csv(
    "adult.data",
    header=None,
    names=COLUMN_NAMES,
    sep=",",
    skipinitialspace=True,
    na_values="?",
)

# adult.test: tem uma linha de cabeçalho estranha ("|1x3 Cross validator")
# que precisa ser pulada, e os rótulos vêm com ponto final (ex: "<=50K.")
df_test = pd.read_csv(
    "adult.test",
    header=None,
    names=COLUMN_NAMES,
    sep=",",
    skipinitialspace=True,
    na_values="?",
    skiprows=1,
)
df_test["income"] = df_test["income"].str.replace(".", "", regex=False)

# Juntamos os dois porque vamos fazer nosso próprio split estratificado
# (mais simples para o trabalho; mantém tudo em um único pipeline)
df = pd.concat([df_train, df_test], ignore_index=True)

print("Shape original (train+test combinados):", df.shape)
print(df.head())

# ---------------------------------------------------------
# 2. LIMPEZA
# ---------------------------------------------------------
obj_cols = df.select_dtypes(include="object").columns
for col in obj_cols:
    df[col] = df[col].astype(str).str.strip()

print("\nValores ausentes por coluna:")
print(df.isna().sum()[df.isna().sum() > 0])

# Remove linhas com valores ausentes (mais simples e defensável para o relatório;
# citem no relatório: alternativa seria imputar pela moda -> "melhoria futura")
linhas_antes = len(df)
df = df.dropna()
print(f"\nLinhas removidas por valores ausentes: {linhas_antes - len(df)}")
print("Shape após remover ausentes:", df.shape)

# Remove duplicatas
duplicatas = df.duplicated().sum()
df = df.drop_duplicates()
print(f"Duplicatas removidas: {duplicatas}")

# 'fnlwgt' é um peso estatístico do censo, sem valor preditivo direto -> remover
df = df.drop(columns=["fnlwgt"])

# ---------------------------------------------------------
# 3. SEPARAR X e y
# ---------------------------------------------------------
y = df["income"].map({"<=50K": 0, ">50K": 1})
X = df.drop(columns=["income"])

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("\nColunas numéricas:", num_cols)
print("Colunas categóricas:", cat_cols)

# ---------------------------------------------------------
# 4. PRÉ-PROCESSAMENTO (normalização + one-hot)
# ---------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ]
)

X_processed = preprocessor.fit_transform(X)

feature_names = num_cols + list(
    preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)
)

df_processed = pd.DataFrame(X_processed, columns=feature_names)
df_processed["income"] = y.values

print("\nShape final pré-processado:", df_processed.shape)

# ---------------------------------------------------------
# 5. TRAIN/TEST SPLIT (estratificado, pois as classes são desbalanceadas ~76%/24%)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y.values,
    test_size=0.2,
    random_state=42,
    stratify=y.values,
)

print("\nDistribuição da classe (treino):", np.bincount(y_train))
print("Distribuição da classe (teste):", np.bincount(y_test))

# ---------------------------------------------------------
# 6. SALVAR ARTEFATOS PARA OS DOIS TRABALHAREM EM PARALELO
# ---------------------------------------------------------
df_processed.to_csv("adult_preprocessado.csv", index=False)

np.savez(
    "train_test_split.npz",
    X_train=X_train, X_test=X_test,
    y_train=y_train, y_test=y_test,
    feature_names=np.array(feature_names),
)

print("\nArquivos gerados: adult_preprocessado.csv e train_test_split.npz")
print("Prontos para a Pessoa A (árvore) e Pessoa B (K-Means) trabalharem em paralelo.")
