"""
PARTE 1 — Árvore de Decisão
Adult Census Income Dataset (UCI)

Pré-requisito: rodar antes preprocessing.py (gera train_test_split.npz)

Requisitos: pip install scikit-learn numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
from sklearn.model_selection import GridSearchCV

# ---------------------------------------------------------
# 1. CARREGAR OS DADOS PRÉ-PROCESSADOS (gerados no Passo 0)
# ---------------------------------------------------------
data = np.load("train_test_split.npz", allow_pickle=True)
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
feature_names = data["feature_names"]

print("X_train:", X_train.shape, " X_test:", X_test.shape)
print("Distribuição treino:", np.bincount(y_train))
print("Distribuição teste:", np.bincount(y_test))


# ---------------------------------------------------------
# Função auxiliar para avaliar e imprimir métricas de um modelo
# ---------------------------------------------------------
def avaliar_modelo(modelo, nome, X_test, y_test):
    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n===== {nome} =====")
    print(f"Acurácia:  {acc:.4f}")
    print(f"Precisão:  {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Matriz de confusão:")
    print(cm)
    print("\nRelatório completo:")
    print(classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))

    return {"nome": nome, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": cm, "modelo": modelo}


# ===========================================================
# MODELO 1 — ÁRVORE SEM LIMITE (baseline, "antes" do ajuste)
# ===========================================================
# Sem restrição de profundidade: tende a decorar o treino (overfitting)
arvore_baseline = DecisionTreeClassifier(random_state=42)
arvore_baseline.fit(X_train, y_train)

resultado_baseline = avaliar_modelo(arvore_baseline, "Árvore SEM poda (baseline)", X_test, y_test)
print(f"Profundidade da árvore: {arvore_baseline.get_depth()}")
print(f"Número de folhas: {arvore_baseline.get_n_leaves()}")

# Acurácia no próprio treino, pra evidenciar overfitting no relatório
acc_treino_baseline = accuracy_score(y_train, arvore_baseline.predict(X_train))
print(f"Acurácia no TREINO (baseline): {acc_treino_baseline:.4f}  <- perto de 1.0 indica overfitting")


# ===========================================================
# MODELO 2 — AJUSTE DE HIPERPARÂMETROS (GridSearchCV)
# ===========================================================
# Testamos combinações de max_depth, min_samples_leaf e criterion.
# Usamos F1-score como métrica de otimização (não acurácia!), porque
# as classes são desbalanceadas (~76% / 24%) e acurácia pode enganar.
param_grid = {
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_leaf": [1, 5, 10, 20],
    "criterion": ["gini", "entropy"],
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

print("\n===== Resultado do GridSearchCV =====")
print("Melhores hiperparâmetros:", grid_search.best_params_)
print(f"Melhor F1 (validação cruzada, treino): {grid_search.best_score_:.4f}")

arvore_ajustada = grid_search.best_estimator_
resultado_ajustada = avaliar_modelo(arvore_ajustada, "Árvore AJUSTADA (pós GridSearch)", X_test, y_test)
print(f"Profundidade da árvore: {arvore_ajustada.get_depth()}")
print(f"Número de folhas: {arvore_ajustada.get_n_leaves()}")

acc_treino_ajustada = accuracy_score(y_train, arvore_ajustada.predict(X_train))
print(f"Acurácia no TREINO (ajustada): {acc_treino_ajustada:.4f}")


# ===========================================================
# COMPARAÇÃO MANUAL DE 3 CONFIGURAÇÕES
# ===========================================================
configuracoes = [
    {"max_depth": None, "min_samples_leaf": 1, "criterion": "gini"},   # = baseline
    {"max_depth": 10, "min_samples_leaf": 5, "criterion": "gini"},
    {"max_depth": 6, "min_samples_leaf": 20, "criterion": "entropy"},
]

print("\n===== Comparação manual de configurações =====")
resultados_configs = []
for cfg in configuracoes:
    modelo = DecisionTreeClassifier(random_state=42, **cfg)
    modelo.fit(X_train, y_train)
    nome = f"depth={cfg['max_depth']}, leaf={cfg['min_samples_leaf']}, crit={cfg['criterion']}"
    r = avaliar_modelo(modelo, nome, X_test, y_test)
    resultados_configs.append(r)


# ===========================================================
# TABELA RESUMO 
# ===========================================================
print("\n===== TABELA RESUMO — Antes vs Depois do ajuste =====")
print(f"{'Modelo':40s} {'Acurácia':>10s} {'Precisão':>10s} {'Recall':>10s} {'F1':>10s}")
for r in [resultado_baseline, resultado_ajustada] + resultados_configs:
    print(f"{r['nome']:40s} {r['acc']:10.4f} {r['prec']:10.4f} {r['rec']:10.4f} {r['f1']:10.4f}")


# ===========================================================
# GRÁFICOS
# ===========================================================
# 1) Matriz de confusão do modelo ajustado
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(resultado_baseline["cm"], display_labels=["<=50K", ">50K"]).plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Baseline (sem poda)")
ConfusionMatrixDisplay(resultado_ajustada["cm"], display_labels=["<=50K", ">50K"]).plot(ax=axes[1], cmap="Blues", colorbar=False)
axes[1].set_title("Ajustada (GridSearch)")
plt.tight_layout()
plt.savefig("matrizes_confusao.png", dpi=150)
print("\nGráfico salvo: matrizes_confusao.png")

# 2) Importância das features (top 15) — bom para discussão no relatório
importancias = arvore_ajustada.feature_importances_
idx_top = np.argsort(importancias)[-15:]
plt.figure(figsize=(8, 6))
plt.barh(range(len(idx_top)), importancias[idx_top])
plt.yticks(range(len(idx_top)), [feature_names[i] for i in idx_top])
plt.xlabel("Importância")
plt.title("Top 15 variáveis mais importantes (árvore ajustada)")
plt.tight_layout()
plt.savefig("importancia_features.png", dpi=150)
print("Gráfico salvo: importancia_features.png")

# 3) Visualização da árvore (limitada a profundidade 3 pra ficar legível)
plt.figure(figsize=(20, 10))
plot_tree(
    arvore_ajustada, max_depth=3, feature_names=feature_names,
    class_names=["<=50K", ">50K"], filled=True, fontsize=8,
)
plt.title("Árvore de decisão (visualização até profundidade 3)")
plt.savefig("arvore_visualizacao.png", dpi=150, bbox_inches="tight")
print("Gráfico salvo: arvore_visualizacao.png")

print("\nConcluído. Use a TABELA RESUMO e os gráficos gerados no relatório.")
