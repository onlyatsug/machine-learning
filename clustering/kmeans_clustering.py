"""
PARTE 2 — Aprendizado Não Supervisionado (K-Means)
Adult Census Income Dataset (UCI)

Pré-requisito: rodar antes o passo0_preprocessing.py
(gera adult_preprocessado.csv). Coloque este script na mesma pasta.

Requisitos: pip install scikit-learn pandas numpy matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------
# 1. CARREGAR OS DADOS PRÉ-PROCESSADOS (gerados no Passo 0)
# ---------------------------------------------------------
df = pd.read_csv("adult_preprocessado.csv")

y = df["income"]                       # guardamos a classe real só para COMPARAR depois
X = df.drop(columns=["income"])        # o K-Means NÃO recebe o rótulo

print("Shape dos dados (sem o rótulo):", X.shape)
print("Distribuição real das classes:", y.value_counts().to_dict())

# ---------------------------------------------------------
# 2. ESCOLHER O NÚMERO DE CLUSTERS (método do cotovelo + silhueta)
# ---------------------------------------------------------
# Rodamos em uma AMOSTRA para o cálculo da silhueta não ficar muito lento
# (o dataset completo tem ~45 mil linhas e 103 colunas).
amostra_idx = np.random.RandomState(42).choice(len(X), size=5000, replace=False)
X_amostra = X.iloc[amostra_idx]

inercias = []
silhuetas = []
k_range = range(2, 8)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_amostra)
    inercias.append(km.inertia_)
    silhuetas.append(silhouette_score(X_amostra, labels))
    print(f"k={k}  inércia={km.inertia_:.1f}  silhueta={silhuetas[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(k_range), inercias, marker="o")
axes[0].set_xlabel("Número de clusters (k)")
axes[0].set_ylabel("Inércia")
axes[0].set_title("Método do cotovelo")

axes[1].plot(list(k_range), silhuetas, marker="o", color="orange")
axes[1].set_xlabel("Número de clusters (k)")
axes[1].set_ylabel("Coeficiente de silhueta")
axes[1].set_title("Coeficiente de silhueta por k")
plt.tight_layout()
plt.savefig("kmeans_escolha_k.png", dpi=150)
print("\nGráfico salvo: kmeans_escolha_k.png")

# ---------------------------------------------------------
# 3. K-MEANS COM K=2 (comparável diretamente com o problema binário)
# ---------------------------------------------------------
# Justificativa: o problema original é binário (<=50K / >50K), então
# k=2 é a escolha mais natural para comparar clusters com classes reais.
K_ESCOLHIDO = 2

kmeans = KMeans(n_clusters=K_ESCOLHIDO, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

print(f"\n===== K-Means com k={K_ESCOLHIDO} (dataset completo) =====")
print("Tamanho de cada cluster:", np.bincount(clusters))

# ---------------------------------------------------------
# 4. COMPARAR CLUSTERS COM AS CLASSES REAIS (matriz de contingência)
# ---------------------------------------------------------
tabela_contingencia = pd.crosstab(
    clusters, y, rownames=["Cluster"], colnames=["Classe real"]
)
print("\nMatriz de contingência (cluster x classe real):")
print(tabela_contingencia)

# Percentual de cada classe dentro de cada cluster (mais fácil de interpretar)
tabela_percentual = pd.crosstab(
    clusters, y, rownames=["Cluster"], colnames=["Classe real"], normalize="index"
) * 100
print("\nPercentual de cada classe dentro de cada cluster:")
print(tabela_percentual.round(1))

tabela_contingencia.to_csv("kmeans_matriz_contingencia.csv")
print("\nTabela salva: kmeans_matriz_contingencia.csv")

# ---------------------------------------------------------
# 5. MÉTRICA DE CONCORDÂNCIA (quanto os clusters se aproximam das classes)
# ---------------------------------------------------------
# Como o K-Means não sabe qual número de cluster corresponde a qual classe,
# testamos as duas correspondências possíveis e ficamos com a melhor.
acc_opcao1 = (clusters == y.values).mean()
acc_opcao2 = (clusters == (1 - y.values)).mean()
concordancia = max(acc_opcao1, acc_opcao2)
print(f"\nConcordância cluster x classe real (melhor correspondência): {concordancia:.4f}")

sil_completo = silhouette_score(X_amostra, KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(X_amostra))
print(f"Coeficiente de silhueta (k=2, amostra): {sil_completo:.4f}")

# ---------------------------------------------------------
# 6. GRÁFICO DOS CLUSTERS (redução de dimensionalidade com PCA para 2D)
# ---------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Painel 1: colorido pelos clusters encontrados pelo K-Means
scatter1 = axes[0].scatter(
    X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", s=5, alpha=0.5
)
axes[0].set_title("Clusters formados pelo K-Means")
axes[0].set_xlabel("Componente principal 1")
axes[0].set_ylabel("Componente principal 2")
legend1 = axes[0].legend(*scatter1.legend_elements(), title="Cluster")
axes[0].add_artist(legend1)

# Painel 2: colorido pelas classes REAIS, para comparação visual direta
scatter2 = axes[1].scatter(
    X_pca[:, 0], X_pca[:, 1], c=y.values, cmap="coolwarm", s=5, alpha=0.5
)
axes[1].set_title("Classes reais (<=50K vs >50K)")
axes[1].set_xlabel("Componente principal 1")
axes[1].set_ylabel("Componente principal 2")
legend2 = axes[1].legend(*scatter2.legend_elements(), title="Classe")
axes[1].add_artist(legend2)

plt.tight_layout()
plt.savefig("kmeans_clusters_pca.png", dpi=150)
print("Gráfico salvo: kmeans_clusters_pca.png")

print(f"\nVariância explicada pelas 2 componentes do PCA: {pca.explained_variance_ratio_.sum()*100:.1f}%")

print("\n===== Concluído. Use a matriz de contingência e os gráficos no relatório. =====")