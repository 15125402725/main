from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (average_precision_score, accuracy_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, precision_recall_curve)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

# Set Chinese display (ensure system has Chinese fonts)
plt.rcParams['font.sans-serif'] = ['SimHei']  # Chinese display for Windows system
plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue

# Create directory to save plots
save_dir = "nystroem_svm_evaluation_plots"
os.makedirs(save_dir, exist_ok=True)

# 1. Data loading
df = pd.read_csv('COUNT_SIS_selected_features.csv')
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# 2. Data split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# 3. Data preprocessing - Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Automatically compute gamma value for RBF kernel (replacing gamma='scale')
gamma_value = 1 / (X_train_scaled.shape[1] * X_train_scaled.var())

# ========== Cross-validation part ==========
print("\n5-fold cross-validation (training set + SMOTE) metrics:")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []
fold_f1_scores = []
fold_roc_aucs = []
fold_avg_precisions = []

for fold, (train_index, val_index) in enumerate(kf.split(X_train_scaled, y_train)):
    X_train_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # SMOTE within each fold
    smote_cv = SMOTE(random_state=42)
    X_train_smote_fold, y_train_smote_fold = smote_cv.fit_resample(X_train_fold, y_train_fold)

    # Build Nystroem SVM model
    model_cv = make_imb_pipeline(
        Nystroem(
            kernel='rbf',
            gamma=gamma_value,
            n_components=100,
            random_state=42
        ),
        SGDClassifier(
            loss='hinge',
            penalty='l2',
            alpha=0.001,
            max_iter=1000,
            tol=1e-3,
            random_state=42,
            class_weight='balanced'
        )
    )

    # Wrap calibrated model to obtain probability outputs
    calibrated_model_cv = CalibratedClassifierCV(model_cv, method='sigmoid', cv=5)
    calibrated_model_cv.fit(X_train_smote_fold, y_train_smote_fold)

    y_pred_cv = calibrated_model_cv.predict(X_val_fold)
    y_proba_cv = calibrated_model_cv.predict_proba(X_val_fold)[:, 1]

    fold_accuracies.append(accuracy_score(y_val_fold, y_pred_cv))
    fold_f1_scores.append(f1_score(y_val_fold, y_pred_cv))
    fold_roc_aucs.append(roc_auc_score(y_val_fold, y_proba_cv))
    fold_avg_precisions.append(average_precision_score(y_val_fold, y_proba_cv))

    print(f"Fold {fold + 1}: Acc={fold_accuracies[-1]:.4f} | F1={fold_f1_scores[-1]:.4f} | "
          f"AUC={fold_roc_aucs[-1]:.4f} | AP={fold_avg_precisions[-1]:.4f}")

print("\nAverage of 5-fold cross-validation:")
print(f"Accuracy: {np.mean(fold_accuracies):.4f} (±{np.std(fold_accuracies):.4f})")
print(f"F1 Score: {np.mean(fold_f1_scores):.4f} (±{np.std(fold_f1_scores):.4f})")
print(f"AUC-ROC: {np.mean(fold_roc_aucs):.4f} (±{np.std(fold_roc_aucs):.4f})")
print(f"Average Precision: {np.mean(fold_avg_precisions):.4f} (±{np.std(fold_avg_precisions):.4f})")

# 5. Build complete pipeline (including SMOTE, Nystroem and SVM) for final model
model = make_imb_pipeline(
    SMOTE(random_state=42),  # SMOTE oversampling
    Nystroem(
        kernel='rbf',
        gamma=gamma_value,  # use computed gamma value
        n_components=100,
        random_state=42
    ),
    SGDClassifier(
        loss='hinge',
        penalty='l2',
        alpha=0.001,
        max_iter=1000,
        tol=1e-3,
        random_state=42,
        class_weight='balanced'
    )
)

# Wrap calibrated model to obtain probability outputs
calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
calibrated_model.fit(X_train_scaled, y_train)  # Note: SMOTE is applied inside each fold of cross-validation

# 6. Prediction
y_pred = calibrated_model.predict(X_test_scaled)
y_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]

# 7. Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
average_precision = average_precision_score(y_test, y_proba)

print("\nModel Evaluation Results (using SMOTE + Nystroem + SVM):")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")
print(f"Average Precision: {average_precision:.4f}")

# Generate timestamp for filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ====================== Visualization plots ======================

# 8.1 Confusion matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Confusion Matrix (Nystroem SVM with SMOTE)')
cm_path = os.path.join(save_dir, f"confusion_matrix_{timestamp}.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()

# 8.2 Metrics bar chart
plt.figure(figsize=(8, 5))
metrics = ['Accuracy', 'F1 Score', 'AUC-ROC', 'Average Precision']
values = [accuracy, f1, roc_auc, average_precision]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = plt.bar(metrics, values, color=colors)
plt.ylim(0, 1.1)
plt.title('Nystroem SVM Model Metrics Comparison (with SMOTE)')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.3f}',
             ha='center', va='bottom')
metrics_path = os.path.join(save_dir, f"metrics_comparison_{timestamp}.png")
plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
plt.close()

# 8.3 Combined ROC and PR curves
plt.figure(figsize=(8, 6))
plt.title('ROC & Precision-Recall Curves (SMOTE_SVM)', fontsize=14, pad=20)

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, color='blue', lw=2,
         label=f'ROC (AUC = {roc_auc:.3f})')

# PR curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.plot(recall, precision, color='red', linestyle='--', lw=2,
         label=f'PR (AP = {average_precision:.3f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall / True Positive Rate', fontsize=12)
plt.ylabel('Precision / Positive Predictive Value', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='lower right', fontsize=12, frameon=False)
combined_curve_path = os.path.join(save_dir, f"combined_curves_{timestamp}.png")
plt.savefig(combined_curve_path, dpi=300, bbox_inches='tight')
plt.close()

# ====================== Output results ======================

# Print key threshold points
print("\nKey threshold point analysis:")
print(f"- When recall = 0.9, precision = {precision[recall >= 0.9][-1]:.2f}")
print(f"- When precision = 0.9, recall = {recall[precision >= 0.9][0]:.2f}")

# Print save paths
print("\nVisualization plots saved to the following paths:")
print(f"- Confusion matrix: {cm_path}")
print(f"- Metrics comparison: {metrics_path}")
print(f"- ROC/PR combined curve: {combined_curve_path}")
