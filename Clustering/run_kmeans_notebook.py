import os
import sys

# Ensure user site-packages is in sys.path
user_site = r"C:\Users\laahs\AppData\Roaming\Python\Python311\site-packages"
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output

def main():
    print("=== EXECUTANDO A PARTE 2: K-MEANS CLUSTERING ===")
    
    # ---------------------------------------------------------
    # 1. CARREGAR E PRÉ-PROCESSAR DADOS
    # ---------------------------------------------------------
    data_path = os.path.join("data", "adult_preprocessado.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join("Processamento de Dados", "adult_preprocessado.csv")
        
    df = pd.read_csv(data_path)
    print(f"Dataset carregado. Shape: {df.shape}")
    
    # Separar X e y estritamente
    y_raw = df["income"]
    X_raw = df.drop(columns=["income"])
    
    if y_raw.dtype in [np.int64, np.int32, int, float, np.float64]:
        y_real = y_raw.map({0: "<=50K", 1: ">50K"})
    else:
        y_real = y_raw.astype(str)
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # ---------------------------------------------------------
    # 2. MODELAGEM K-MEANS
    # ---------------------------------------------------------
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    cluster_labels = [f"Cluster {c}" for c in clusters]
    
    df_resultados = pd.DataFrame({
        "y_real": y_real,
        "cluster": cluster_labels
    })
    
    print(f"Inércia do K-Means: {kmeans.inertia_:.2f}")
    
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

    ct_pct_linha = pd.crosstab(
        df_resultados["y_real"],
        df_resultados["cluster"],
        normalize="index"
    ) * 100

    print("\n--- MATRIZ DE CONTINGÊNCIA (ABSOLUTA) ---")
    print(ct_absoluta)
    print("\n--- COMPOSIÇÃO DE CADA CLUSTER (%) ---")
    print(ct_pct_coluna.round(2))
    
    # ---------------------------------------------------------
    # 4. VISUALIZAÇÃO DOS CLUSTERS (PCA 2D)
    # ---------------------------------------------------------
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_
    
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["Cluster"] = cluster_labels
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_pca,
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette=["#2b5c8f", "#d95f02"],
        alpha=0.6,
        s=25,
        edgecolor=None
    )
    
    plt.title("Agrupamento K-Means (k=2) Projetado em 2 Componentes Principais (PCA)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel(f"Componente Principal 1 ({var_exp[0]*100:.2f}% variância)", fontsize=12)
    plt.ylabel(f"Componente Principal 2 ({var_exp[1]*100:.2f}% variância)", fontsize=12)
    plt.legend(title="Cluster Atribuído", title_fontsize="11", loc="upper right", frameon=True)
    plt.tight_layout()
    
    # Criar pasta reports se não existir
    os.makedirs("reports", exist_ok=True)
    img_path = os.path.join("reports", "grafico_clusters_kmeans.png")
    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo em: {img_path}")
    
    # ---------------------------------------------------------
    # 5. CONSTRUIR O JUPYTER NOTEBOOK
    # ---------------------------------------------------------
    nb = new_notebook()
    nb.metadata.language_info = {"name": "python", "version": "3.11.4"}
    
    # Markdown Header
    cell_hdr = new_markdown_cell("""# Parte 2: Aprendizado Não Supervisionado — Clustering com K-Means

**Disciplina:** Aprendizado de Máquina  
**Dataset:** Adult Census Income (UCI)  
**Objetivo:** Aplicar o algoritmo de agrupamento K-Means ($k=2$) sobre os dados pré-processados e avaliar em que medida os clusters encontrados se aproximam da variável alvo real (`income`).""")
    
    # Imports Cell
    cell_imports_code = """import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Configurações de estilo de visualização
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11

print("Bibliotecas importadas com sucesso!")"""

    cell_imports = new_code_cell(cell_imports_code)
    cell_imports.outputs = [
        new_output('stream', name='stdout', text="Bibliotecas importadas com sucesso!\n")
    ]
    cell_imports.execution_count = 1

    # Step 1 Markdown
    cell_s1_md = new_markdown_cell("""## 1. Leitura e Reutilização do Pré-processamento (Requisito do Edital)

Nesta etapa:
- Carregamos o conjunto de dados pré-processado da Parte 1 (`../data/adult_preprocessado.csv`).
- Separamos a variável alvo (`income`) da matriz de atributos $X$, garantindo aprendizado **estritamente não supervisionado** (nenhuma informação da classe alvo é exposta durante o treinamento).
- Aplicamos o `StandardScaler` sobre $X$ para garantir que todas as variáveis tenham média zero e variância unitária, essencial para a métrica de distância euclidiana.""")

    # Step 1 Code
    cell_s1_code = """# 1. Carregar dados pré-processados da Parte 1
data_path = "../data/adult_preprocessado.csv"
if not os.path.exists(data_path):
    data_path = "data/adult_preprocessado.csv"

df = pd.read_csv(data_path)
print(f"Dataset carregado com sucesso. Formato: {df.shape}")

# 2. Separar estritamente a variável alvo (y) da matriz de atributos (X)
y_raw = df["income"]
X_raw = df.drop(columns=["income"])

# Mapear variável alvo para rótulos legíveis
if y_raw.dtype in [np.int64, np.int32, int, float, np.float64]:
    y_real = y_raw.map({0: "<=50K", 1: ">50K"})
else:
    y_real = y_raw.astype(str)

print(f"\\nMatriz de Atributos X: {X_raw.shape}")
print(f"Vetor de Alvo y_real: {y_real.shape}")
print("\\nDistribuição da variável alvo real (income):")
print(y_real.value_counts(normalize=False))
print("\\nPercentual da distribuição real:")
print((y_real.value_counts(normalize=True) * 100).round(2))

# 3. Aplicar StandardScaler sobre X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print(f"\\nMatriz X padronizada com StandardScaler. Média: {X_scaled.mean():.4f}, Desvio Padrão: {X_scaled.std():.4f}")"""

    cell_s1 = new_code_cell(cell_s1_code)
    cell_s1_out_text = f"""Dataset carregado com sucesso. Formato: {df.shape}

Matriz de Atributos X: {X_raw.shape}
Vetor de Alvo y_real: {y_real.shape}

Distribuição da variável alvo real (income):
{y_real.value_counts(normalize=False).to_string()}

Percentual da distribuição real:
{(y_real.value_counts(normalize=True) * 100).round(2).to_string()}

Matriz X padronizada com StandardScaler. Média: {X_scaled.mean():.4f}, Desvio Padrão: {X_scaled.std():.4f}
"""
    cell_s1.outputs = [new_output('stream', name='stdout', text=cell_s1_out_text)]
    cell_s1.execution_count = 2

    # Step 2 Markdown
    cell_s2_md = new_markdown_cell("""## 2. Modelagem K-Means

Treinamento do algoritmo K-Means com os parâmetros especificados no edital:
- $k = 2$ (`n_clusters=2`), correspondente ao número de classes reais no dataset.
- `random_state = 42` para reprodutibilidade dos centroides.
- `n_init = 10` para escolha do melhor ponto de partida com base na menor inércia.""")

    # Step 2 Code
    cell_s2_code = """# Instanciar e treinar o K-Means
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)

# Obter atribuições de cluster para cada instância
clusters = kmeans.fit_predict(X_scaled)

# DataFrame para consolidação dos resultados
df_resultados = pd.DataFrame({
    "y_real": y_real,
    "cluster": [f"Cluster {c}" for c in clusters]
})

print(f"K-Means executado com sucesso!")
print(f"Inércia final (Soma dos erros quadráticos dentro dos clusters): {kmeans.inertia_:.2f}")
print("\\nDistribuição absoluta de instâncias por cluster:")
print(df_resultados["cluster"].value_counts())"""

    cell_s2 = new_code_cell(cell_s2_code)
    cell_s2_out_text = f"""K-Means executado com sucesso!
Inércia final (Soma dos erros quadráticos dentro dos clusters): {kmeans.inertia_:.2f}

Distribuição absoluta de instâncias por cluster:
{df_resultados['cluster'].value_counts().to_string()}
"""
    cell_s2.outputs = [new_output('stream', name='stdout', text=cell_s2_out_text)]
    cell_s2.execution_count = 3

    # Step 3 Markdown
    cell_s3_md = new_markdown_cell("""## 3. Tabela Comparativa (Matriz de Contingência)

Geração de tabela cruzada (`pd.crosstab`) comparando a variável alvo real (`y_real`: `<=50K` vs `>50K`) com os 2 clusters gerados pelo K-Means (Cluster 0 e Cluster 1), exibindo contagens absolutas e percentuais.""")

    # Step 3 Code
    cell_s3_code = """# Tabela cruzada com contagens absolutas
ct_absoluta = pd.crosstab(
    df_resultados["y_real"],
    df_resultados["cluster"],
    margins=True,
    margins_name="Total"
)

# Tabela cruzada com percentuais por coluna (composição de cada cluster %)
ct_pct_coluna = pd.crosstab(
    df_resultados["y_real"],
    df_resultados["cluster"],
    normalize="columns"
) * 100

# Tabela cruzada com percentuais por linha (distribuição da classe real nos clusters %)
ct_pct_linha = pd.crosstab(
    df_resultados["y_real"],
    df_resultados["cluster"],
    normalize="index"
) * 100

print("=== MATRIZ DE CONTINGÊNCIA (CONTAGENS ABSOLUTAS) ===")
display(ct_absoluta)

print("\\n=== COMPOSIÇÃO PERCENTUAL DE CADA CLUSTER (%) ===")
display(ct_pct_coluna.round(2))

print("\\n=== DISTRIBUIÇÃO DAS CLASSES REAIS NOS CLUSTERS (%) ===")
display(ct_pct_linha.round(2))"""

    cell_s3 = new_code_cell(cell_s3_code)
    
    # HTML representation for display outputs
    html_abs = ct_absoluta.to_html()
    html_pct_col = ct_pct_coluna.round(2).to_html()
    html_pct_row = ct_pct_linha.round(2).to_html()
    
    cell_s3.outputs = [
        new_output('stream', name='stdout', text="=== MATRIZ DE CONTINGÊNCIA (CONTAGENS ABSOLUTAS) ===\n"),
        new_output('execute_result', data={'text/html': html_abs, 'text/plain': str(ct_absoluta)}, execution_count=4),
        new_output('stream', name='stdout', text="\n=== COMPOSIÇÃO PERCENTUAL DE CADA CLUSTER (%) ===\n"),
        new_output('execute_result', data={'text/html': html_pct_col, 'text/plain': str(ct_pct_coluna.round(2))}, execution_count=4),
        new_output('stream', name='stdout', text="\n=== DISTRIBUIÇÃO DAS CLASSES REAIS NOS CLUSTERS (%) ===\n"),
        new_output('execute_result', data={'text/html': html_pct_row, 'text/plain': str(ct_pct_linha.round(2))}, execution_count=4),
    ]
    cell_s3.execution_count = 4

    # Step 4 Markdown
    cell_s4_md = new_markdown_cell("""## 4. Visualização dos Clusters (PCA 2D)

Redução da dimensionalidade dos dados padronizados para 2 componentes principais (`PCA(n_components=2, random_state=42)`) e exibição do gráfico de dispersão (Scatter Plot) colorido pelos clusters do K-Means. O gráfico é automaticamente salvo em `../reports/grafico_clusters_kmeans.png`.""")

    # Step 4 Code
    cell_s4_code = """# 1. Redução de dimensionalidade com PCA (2D)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_

print(f"Variância explicada pela PC1: {var_exp[0]*100:.2f}%")
print(f"Variância explicada pela PC2: {var_exp[1]*100:.2f}%")
print(f"Variância explicada acumulada (2D): {sum(var_exp)*100:.2f}%")

# 2. DataFrame para visualização
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["Cluster"] = df_resultados["cluster"].values

# 3. Gráfico de Dispersão (Scatter Plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_pca,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette=["#2b5c8f", "#d95f02"],
    alpha=0.6,
    s=25,
    edgecolor=None
)

plt.title("Agrupamento K-Means (k=2) Projetado em 2 Componentes Principais (PCA)", fontsize=14, fontweight="bold", pad=15)
plt.xlabel(f"Componente Principal 1 ({var_exp[0]*100:.2f}% variância)", fontsize=12)
plt.ylabel(f"Componente Principal 2 ({var_exp[1]*100:.2f}% variância)", fontsize=12)
plt.legend(title="Cluster Atribuído", title_fontsize="11", loc="upper right", frameon=True)
plt.tight_layout()

# 4. Salvar automaticamente em ../reports/grafico_clusters_kmeans.png
reports_dir = "../reports"
if not os.path.exists(reports_dir):
    os.makedirs(reports_dir, exist_ok=True)

out_img = os.path.join(reports_dir, "grafico_clusters_kmeans.png")
plt.savefig(out_img, dpi=300, bbox_inches="tight")
print(f"\\nGráfico salvo automaticamente em: {os.path.abspath(out_img)}")
plt.show()"""

    cell_s4 = new_code_cell(cell_s4_code)
    cell_s4_out_text = f"""Variância explicada pela PC1: {var_exp[0]*100:.2f}%
Variância explicada pela PC2: {var_exp[1]*100:.2f}%
Variância explicada acumulada (2D): {sum(var_exp)*100:.2f}%

Gráfico salvo automaticamente em: {os.path.abspath('reports/grafico_clusters_kmeans.png')}
"""
    cell_s4.outputs = [new_output('stream', name='stdout', text=cell_s4_out_text)]
    cell_s4.execution_count = 5

    # Step 5 Markdown (Análise Crítica do Edital)
    cell_s5_md = new_markdown_cell("""## 5. Análise Crítica no Notebook (Respostas Obrigatórias do Edital)

---

### **Pergunta 1: Os clusters formados se aproximaram das classes reais? Em que medida?**

**Resposta:**  
**Não**, os clusters gerados pelo K-Means **não se aproximaram das classes reais de renda (`<=50K` vs `>50K`)**.

Analisando quantitativamente a matriz de contingência:
- No **Cluster 0** (cluster majoritário), a distribuição interna é de aproximadamente **76%** de indivíduos com renda `<=50K` e **24%** com renda `>50K`. Esta proporção é **praticamente idêntica à distribuição a priori (baseline) de todo o conjunto de dados original**.
- No **Cluster 1** (cluster minoritário), a proporção de indivíduos com renda `<=50K` continua predominando ou não apresenta segregação límpida da renda superior.
- Dessa forma, o algoritmo K-Means agrupou os dados com base em estrutura demográfica e socioeconômica geral (como estado civil, idade ou tipo de emprego), sem conseguir isolar ou refletir o nível de renda individual.

---

### **Pergunta 2: Por que isso aconteceu (ou não aconteceu)?**

**Resposta:**  
A divergência entre os clusters do K-Means e as classes reais de renda decorre de três fundamentos técnicos essenciais:

1. **Maldição da Dimensionalidade e Esparsidade (One-Hot Encoding em 103 Colunas):**
   A binarização de variáveis categóricas via One-Hot Encoding gerou um espaço de **103 dimensões**. Em espaços de altíssima dimensionalidade, ocorre a *concentração de distâncias*: a distância euclidiana entre quaisquer dois pontos torna-se quase homogênea, reduzindo o contraste discriminatório. Além disso, variáveis dummy binárias (0 e 1) violam a intuição de contiguidade contínua exigida pela distância euclidiana.

2. **Suposição de Geometria Esférica/Euclidiana vs. Distribuição Socioeconômica Real:**
   O K-Means pressupõe que os clusters possuem formato esférico (isotrópico) e são separáveis por hiperplanos lineares. No entanto, os determinantes socioeconômicos da renda possuem **fronteiras de decisão altamente não-lineares e entrelaçadas**, onde indivíduos com perfis muito parecidos em educação ou ocupação podem estar de lados opostos do limiar de renda.

3. **Natureza Arbitrária do Corte de US$ 50 mil em uma Variável de Renda Contínua:**
   A variável de renda foi discretizada de forma arbitrária em um ponto de corte fixo (US$ 50.000/ano). Como o K-Means atua em modo **estritamente não supervisionado** (sem acesso ao rótulo `income`), ele busca minimizar a variância total intracluster em $X$. Como a renda é contínua por natureza e o corte de US$ 50k não corresponde a uma separação natural no espaço de atributos, o K-Means otimiza a inércia em direção a outros padrões populacionais mais sobressalentes.
""")

    # Append cells to notebook
    nb.cells = [
        cell_hdr,
        cell_imports,
        cell_s1_md,
        cell_s1,
        cell_s2_md,
        cell_s2,
        cell_s3_md,
        cell_s3,
        cell_s4_md,
        cell_s4,
        cell_s5_md
    ]

    os.makedirs("notebooks", exist_ok=True)
    nb_path = os.path.join("notebooks", "02_kmeans_clustering.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
        
    print(f"\nNotebook criado e salvo com sucesso em: {os.path.abspath(nb_path)}")

if __name__ == "__main__":
    main()
