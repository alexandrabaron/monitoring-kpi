import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve

# ----- Changement global de police -----
plt.rcParams['font.family'] = 'Segoe UI'

# ----- Palette de couleurs -----
colors = [
    (92/255, 138/255, 148/255),
    (161/255, 196/255, 56/255),
    (0/255, 163/255, 140/255),
    (186/255, 61/255, 115/255),
    (253/255, 149/255, 83/255)
]

# 1. Création d'un dataset simple
np.random.seed(0)
n_points = 200
X = np.random.randn(n_points)
y = (X + np.random.randn(n_points) * 0.5 > 0).astype(int)

X = X.reshape(-1, 1)

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Entrainement du modèle
model = LogisticRegression()
model.fit(X_train, y_train)

# Prédiction des probabilités
y_probs = model.predict_proba(X_test)[:, 1]

# 3. Visualisation du dataset
plt.figure(figsize=(6, 4))
plt.scatter(X_train, y_train, c=[colors[i] for i in y_train], alpha=0.7, edgecolors='k')
plt.title("Dataset d'entraînement")
plt.xlabel("X")
plt.ylabel("Classe")
plt.grid()
plt.savefig('monitoring-kpi/roc auc/img/1_dataset_entraînement.png')

# 4. Courbe de probabilité + échantillons colorés
x_range = np.linspace(X.min(), X.max(), 300).reshape(-1,1)
y_pred_prob = model.predict_proba(x_range)[:,1]

plt.figure(figsize=(8, 5))

# Courbe de prédiction continue
plt.plot(x_range, y_pred_prob, color=colors[0], label='Probabilité prédite (régression logistique)')

# Ajout des échantillons
plt.scatter(X_test, y_test, c=[colors[i] for i in y_test], edgecolors='k', label='Échantillons', alpha=0.8)

# Seuil de décision
plt.axhline(0.5, color=colors[3], linestyle='--', label='Seuil 0.5')

plt.title("Régression logistique et échantillons test")
plt.xlabel("X")
plt.ylabel("Probabilité / Classe")
plt.legend()
plt.grid()
plt.savefig('monitoring-kpi/roc auc/img/2_reg.png')

# 5. Influence du seuil sur la matrice de confusion
seuils = [0.3, 0.5, 0.7]
fig, axes = plt.subplots(1, len(seuils), figsize=(15,4))

for idx, seuil in enumerate(seuils):
    y_pred = (y_probs >= seuil).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    axes[idx].imshow(cm, cmap='Blues')
    axes[idx].set_title(f"Seuil = {seuil}")
    for i in range(2):
        for j in range(2):
            axes[idx].text(j, i, cm[i, j], ha='center', va='center', color='black')
    axes[idx].set_xlabel('Prédit')
    axes[idx].set_ylabel('Vrai')

plt.tight_layout()
plt.savefig('monitoring-kpi/roc auc/img/3_influ_seuil.png')

# 6. Courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = roc_auc_score(y_test, y_probs)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color=colors[1], label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], 'k--', label="Modèle random")
plt.xlabel('Taux de faux positifs (FPR)')
plt.ylabel('Taux de vrais positifs (TPR)')
plt.title('Courbe ROC')
plt.legend(loc="lower right")
plt.grid()
plt.savefig('monitoring-kpi/roc auc/img/4_courbe_roc.png')

# 8. Comparaison plusieurs modèles
perfect_probs = y_test.copy()
random_probs = np.random.rand(len(y_test))

fpr_model, tpr_model, _ = roc_curve(y_test, y_probs)
fpr_perfect, tpr_perfect, _ = roc_curve(y_test, perfect_probs)
fpr_random, tpr_random, _ = roc_curve(y_test, random_probs)

plt.figure(figsize=(7, 6))
plt.plot(fpr_model, tpr_model, color=colors[0], label=f"Notre modèle (AUC={roc_auc_score(y_test, y_probs):.2f})")
plt.plot(fpr_perfect, tpr_perfect, color=colors[2], label="Modèle parfait (AUC=1.0)")
plt.plot(fpr_random, tpr_random, color=colors[3], label=f"Modèle aléatoire (AUC={roc_auc_score(y_test, random_probs):.2f})")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Comparaison des courbes ROC')
plt.legend()
plt.grid()
plt.savefig('monitoring-kpi/roc auc/img/5_comp_roc.png')

# 9. Courbe Precision-Recall
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_probs)

plt.figure(figsize=(6,4))
plt.plot(recall, precision, color=colors[4], label="Precision-Recall curve")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Courbe Precision-Recall')
plt.grid()
plt.legend()
plt.savefig('monitoring-kpi/roc auc/img/6_precision_recall.png')
