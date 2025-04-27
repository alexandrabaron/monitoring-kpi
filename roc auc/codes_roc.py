import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve

# 1. Création d'un dataset simple
np.random.seed(0)
n_points = 200
X = np.random.randn(n_points)
y = (X + np.random.randn(n_points) * 0.5 > 0).astype(int)  # Proba de 0 ou 1

X = X.reshape(-1, 1)

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Entrainement du modèle
model = LogisticRegression()
model.fit(X_train, y_train)

# Prédiction des probabilités
y_probs = model.predict_proba(X_test)[:, 1]  # Proba d'appartenir à la classe 1

# 3. Visualisation du dataset
plt.figure(figsize=(6, 4))
plt.scatter(X_train, y_train, c=y_train, cmap='bwr', alpha=0.7, edgecolors='k')
plt.title("Dataset d'entraînement")
plt.xlabel("X")
plt.ylabel("Classe")
plt.grid()
plt.savefig('img/dataset_entraînement.png')  # <-- Sauvegarde ici
plt.show()

# 4. Courbe de probabilité + seuil 0.5
x_range = np.linspace(X.min(), X.max(), 300).reshape(-1,1)
y_pred_prob = model.predict_proba(x_range)[:,1]

# 4. Courbe de probabilité + échantillons colorés
plt.figure(figsize=(8, 5))

# Courbe de prédiction continue
plt.plot(x_range, y_pred_prob, color='black', label='Probabilité prédite (régression logistique)')

# Ajout des échantillons
plt.scatter(X_test, y_test, c=y_test, cmap='bwr', edgecolors='k', label='Échantillons', alpha=0.8)

# Seuil de décision
plt.axhline(0.5, color='red', linestyle='--', label='Seuil 0.5')

plt.title("Régression logistique et échantillons test")
plt.xlabel("X")
plt.ylabel("Probabilité / Classe")
plt.legend()
plt.grid()
plt.savefig('img/fig2_regression_et_points.png', dpi=300)
plt.show()


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
plt.show()

# 6. Courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = roc_auc_score(y_test, y_probs)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], 'k--', label="Modèle random")
plt.xlabel('Taux de faux positifs (FPR)')
plt.ylabel('Taux de vrais positifs (TPR)')
plt.title('Courbe ROC')
plt.legend(loc="lower right")
plt.grid()
plt.show()

# 7. Visualisation AUC (optionnelle car déjà vue sur la ROC)

# 8. Comparaison plusieurs modèles (exemple fictif ici)
# Modèle parfait
perfect_probs = y_test.copy()

# Modèle aléatoire
random_probs = np.random.rand(len(y_test))

fpr_model, tpr_model, _ = roc_curve(y_test, y_probs)
fpr_perfect, tpr_perfect, _ = roc_curve(y_test, perfect_probs)
fpr_random, tpr_random, _ = roc_curve(y_test, random_probs)

plt.figure(figsize=(7, 6))
plt.plot(fpr_model, tpr_model, label=f"Notre modèle (AUC={roc_auc_score(y_test, y_probs):.2f})")
plt.plot(fpr_perfect, tpr_perfect, label="Modèle parfait (AUC=1.0)")
plt.plot(fpr_random, tpr_random, label=f"Modèle aléatoire (AUC={roc_auc_score(y_test, random_probs):.2f})")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Comparaison des courbes ROC')
plt.legend()
plt.grid()
plt.show()

# 9. (Bonus) Courbe Precision-Recall
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_probs)

plt.figure(figsize=(6,4))
plt.plot(recall, precision, label="Precision-Recall curve")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Courbe Precision-Recall')
plt.grid()
plt.legend()
plt.show()
