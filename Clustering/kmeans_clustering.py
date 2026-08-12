"""
PARTE 2 — Aprendizado Não Supervisionado (K-Means)
Adult Census Income Dataset (UCI)

Pré-requisito: ter o arquivo adult_preprocessado.csv na pasta data/ (ou no mesmo diretório)

Requisitos: pip install scikit-learn numpy pandas matplotlib seaborn
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ---------------------------------------------------------
# 1. CARREGAR E PRÉ-PROCESSAR OS DADOS (Reutilização da Parte 1)
# ---------------------------------------------------------
# Tenta encontrar o arquivo pré-processado nas pastas usuais
caminho_dados = "data/adult_preprocessado.csv"
if not os.path.exists(caminho_dados):
    caminho_dados = "adult_preprocessado.csv"
if not os.path.exists(caminho_dados):
    caminho_dados = "Clustering/adult_preprocessado.csv"
if not os.path.exists(caminho_dados):
    caminho_dados = "../Processamento de Dados/adult_preprocessado.csv"
if not os.path.exists(caminho_dados):
    caminho_dados = "Processamento de Dados/adult_preprocessado.csv"

df = pd.read_csv(caminho_dados)
print(f"Dataset carregado com sucesso. Formato: {df.shape}")

# Separar ESTRITAMENTE a variável alvo (y) da matriz de atributos (X)
# Garantia de aprendizado não supervisionado: y NUNCA entra no K-Means
y_raw = df["income"]
X_raw = df.drop(columns=["income"])

# Converter rótulos numéricos para texto legível (se necessário)
if y_raw.dtype in [np.int64, np.int32, int, float, np.float64]:
    y_real = y_raw.map({0: "<=50K", 1: ">50K"})
else:
    y_real = y_raw.astype(str)

print("Matriz X (atributos):", X_raw.shape)
print("Distribuição real da classe alvo:")
print(y_real.value_counts())

# Padronização contínua com StandardScaler (crucial para distância euclidiana)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)


# ---------------------------------------------------------
# 2. MODELAGEM K-MEANS (k = 2)
# ---------------------------------------------------------
# Treinamento com k=2 (duas classes de renda) e semente fixa para reprodutibilidade
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df_resultados = pd.DataFrame({
    "y_real": y_real,
    "cluster": [f"Cluster {c}" for c in clusters]
})

print("\n===== RESULTADOS DO K-MEANS =====")
print(f"Inércia final (Soma dos erros quadráticos): {kmeans.inertia_:.2f}")
print("Distribuição das instâncias por cluster:")
print(df_resultados["cluster"].value_counts())


# ---------------------------------------------------------
# 3. TABELA COMPARATIVA (MATRIZ DE CONTINGÊNCIA)
# ---------------------------------------------------------
ct_absoluta = pd.crosstab(
    df_resultados["y_real"],
    df_resultados["cluster"],
    margins=True,
    margins_name="Total"
)

ct_pct_coluna = pd.crosstab(
    df_resultados["y_real"],
    df_resultados["cluster"],
    normalize="columns"
) * 100

print("\n===== MATRIZ DE CONTINGÊNCIA (ABSOLUTA) =====")
print(ct_absoluta)

print("\n===== COMPOSIÇÃO PERCENTUAL DE CADA CLUSTER (%) =====")
print(ct_pct_coluna.round(2))


# ---------------------------------------------------------
# 4. GERAÇÃO DE GRÁFICOS (ARTEFATOS PARA O RELATÓRIO)
# ---------------------------------------------------------
# 1) Gráfico de Dispersão PCA 2D (EXIGÊNCIA OBRIGATÓRIA DO EDITAL)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_

df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["Cluster"] = df_resultados["cluster"].values

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df_pca,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette=["#1f77b4", "#ff7f0e"],
    alpha=0.5,
    s=20,
    edgecolor=None
)
plt.title("Agrupamento K-Means (k=2) via PCA 2D", fontsize=12, fontweight="bold")
plt.xlabel(f"Componente Principal 1 ({var_exp[0]*100:.2f}% variância)")
plt.ylabel(f"Componente Principal 2 ({var_exp[1]*100:.2f}% variância)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("grafico_clusters_kmeans.png", dpi=150)
print("\nGráfico salvo: grafico_clusters_kmeans.png")

# 2) Heatmap da Matriz de Contingência (OPCIONAL/RECOMENDADO)
ct_sem_total = pd.crosstab(df_resultados["y_real"], df_resultados["cluster"])
plt.figure(figsize=(6, 5))
sns.heatmap(
    ct_sem_total,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    linewidths=1,
    linecolor="white",
    annot_kws={"size": 12, "weight": "bold"}
)
plt.title("Matriz de Contingência: Classe Real vs Cluster", fontsize=11, fontweight="bold")
plt.xlabel("Cluster K-Means")
plt.ylabel("Renda Real (income)")
plt.tight_layout()
plt.savefig("matriz_contingencia_heatmap.png", dpi=150)
print("Gráfico salvo: matriz_contingencia_heatmap.png")


# ---------------------------------------------------------
# 5. ANÁLISE CRÍTICA (RESPOSTAS OBRIGATÓRIAS DO EDITAL)
# ---------------------------------------------------------
print("\n" + "="*60)
print("ANÁLISE CRÍTICA PARA O RELATÓRIO TEÓRICO")
print("="*60)
print("""
1. Os clusters se aproximaram das classes reais?
   NÃO. O K-Means não conseguiu separar a renda (<=50K vs >50K). 
   Ambos os clusters contêm uma proporção misturada das duas classes de renda.

2. Por que isso aconteceu?
   a) Maldição da Dimensionalidade: O One-Hot Encoding gerou muitas colunas 
      esparsas, onde a distância euclidiana perde capacidade de diferenciação.
   b) Geometria dos Dados: O K-Means busca agrupamentos esféricos lineares, 
      enquanto a separação de renda é altamente não-linear e complexa.
   c) Ausência do Rótulo: O algoritmo minimizou a inércia geral das variáveis 
      demográficas e profissionais, agrupando padrões populacionais mais fortes 
      do que o corte arbitrário da classe de renda.
""")
print("Concluído. Use a Matriz de Contingência e os gráficos gerados no relatório.")
