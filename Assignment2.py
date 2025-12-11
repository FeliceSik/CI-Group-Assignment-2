import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score
)
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras import layers, models

np.set_printoptions(precision=3, suppress=True)

# ===========================
# TRAIN oversampling target:
# True count will become TARGET_RATIO * False count
# Example: 0.9 means True ~ 90% of False
# ===========================
TARGET_RATIO = 0.8
# ===========================


# Helper: dense conversion
def to_dense(Xm):
    return Xm.toarray() if hasattr(Xm, "toarray") else Xm

# Helper: plot confusion matrix (matplotlib only)
def plot_confusion(cm, title):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()


# Build model (same as yours)
def build_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy")
        ]
    )
    return model


# ============================================================
# 1) Load dataset
# ============================================================
data_path = "online_shoppers_intention.csv"
df = pd.read_csv(data_path)

print("Shape of dataset:", df.shape)
print("\nColumns:", df.columns)
print("\nFirst 5 rows:")
print(df.head())

# Original distribution plot (always imbalanced)
print("\nRevenue value counts (ORIGINAL DATA):")
print(df["Revenue"].value_counts())
print("\nRevenue proportion (ORIGINAL DATA):")
print(df["Revenue"].value_counts(normalize=True))

plt.figure()
df["Revenue"].value_counts().plot(kind="bar")
plt.title("Class Distribution of Revenue (ORIGINAL DATA)")
plt.xticks(rotation=0)
plt.xlabel("Revenue")
plt.ylabel("Count")
plt.show()

# Features/target
X = df.drop("Revenue", axis=1)
y = df["Revenue"].astype(int)

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 2) Train / Val / Test split
# ============================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("\nTrain size:", X_train.shape[0])
print("Val size:", X_val.shape[0])
print("Test size:", X_test.shape[0])

print("\nTrain Revenue counts (BEFORE oversampling):")
print(y_train.value_counts())

# IMPORTANT: save the pre-oversampling train split for the ratio search later
X_train_base = X_train.copy()
y_train_base = y_train.copy()


# ============================================================
# 3) Oversampling TRAIN ONLY (to chosen TARGET_RATIO)
# ============================================================
train_df = X_train.copy()
train_df["Revenue"] = y_train.values

df_majority = train_df[train_df["Revenue"] == 0]
df_minority = train_df[train_df["Revenue"] == 1]

N0 = len(df_majority)
target_N1 = int(TARGET_RATIO * N0)

df_minority_over = resample(
    df_minority,
    replace=True,
    n_samples=target_N1,
    random_state=42
)

train_balanced = pd.concat([df_majority, df_minority_over]).sample(frac=1, random_state=42)

X_train = train_balanced.drop("Revenue", axis=1)
y_train = train_balanced["Revenue"].astype(int)

print(f"\nTARGET_RATIO = {TARGET_RATIO}")
print("Train Revenue counts (AFTER oversampling):")
print(y_train.value_counts())

plt.figure()
y_train.value_counts().plot(kind="bar")
plt.title(f"Revenue Distribution (TRAIN after oversampling, ratio={TARGET_RATIO})")
plt.xticks(rotation=0)
plt.xlabel("Revenue")
plt.ylabel("Count")
plt.show()


# ============================================================
# 4) Preprocessing + PCA
# ============================================================
categorical_cols = ["Month", "VisitorType", "Weekend"]
numeric_cols = [c for c in X.columns if c not in categorical_cols]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc = preprocessor.transform(X_val)
X_test_proc = preprocessor.transform(X_test)

X_train_dense = to_dense(X_train_proc)
X_val_dense = to_dense(X_val_proc)
X_test_dense = to_dense(X_test_proc)

pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_dense)
X_val_pca = pca.transform(X_val_dense)
X_test_pca = pca.transform(X_test_dense)

input_dim = X_train_pca.shape[1]


# ============================================================
# 5) Class weights (kept as you had it)
# ============================================================
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = {cls: w for cls, w in zip(classes, class_weights)}
print("\nClass weights:", class_weight_dict)


# ============================================================
# 6) Train model
# ============================================================
model = build_model(input_dim)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc",
    patience=5,
    mode="max",
    restore_best_weights=True
)

history = model.fit(
    X_train_pca, y_train,
    validation_data=(X_val_pca, y_val),
    epochs=50,
    batch_size=64,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1
)

history_dict = history.history

# Curves
plt.figure()
plt.plot(history_dict["auc"], label="Train AUC")
plt.plot(history_dict["val_auc"], label="Val AUC")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("Training vs Validation AUC")
plt.legend()
plt.show()

plt.figure()
plt.plot(history_dict["loss"], label="Train loss")
plt.plot(history_dict["val_loss"], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()

if "accuracy" in history_dict and "val_accuracy" in history_dict:
    plt.figure()
    plt.plot(history_dict["accuracy"], label="Train accuracy")
    plt.plot(history_dict["val_accuracy"], label="Val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.show()


# ============================================================
# 7) Pick threshold that maximizes F1 on validation
# ============================================================
val_prob = model.predict(X_val_pca).ravel()
thr_grid = np.linspace(0.0, 1.0, 1001)

val_f1s = []
for t in thr_grid:
    val_pred = (val_prob >= t).astype(int)
    val_f1s.append(f1_score(y_val, val_pred, zero_division=0))

best_idx = int(np.argmax(val_f1s))
best_threshold = float(thr_grid[best_idx])

print(f"\nBest threshold (max F1 on validation): {best_threshold:.3f}")
print(f"Best validation F1: {val_f1s[best_idx]:.4f}")


# ============================================================
# 8) Reports + Confusion matrices + ROC
# ============================================================
# TRAIN report (after oversampling)
train_prob = model.predict(X_train_pca).ravel()
train_pred = (train_prob >= best_threshold).astype(int)

print("\n==============================")
print("Classification Report (TRAIN after oversampling)")
print("(Optimistic; for analysis only)")
print("==============================")
print(classification_report(y_train, train_pred, digits=3))

cm_train = confusion_matrix(y_train, train_pred)
print("Confusion Matrix (TRAIN):")
print(cm_train)
plot_confusion(cm_train, "Confusion Matrix (TRAIN after oversampling)")

# TEST report (official)
test_results = model.evaluate(X_test_pca, y_test, verbose=0)
print("\nKeras test results:")
for name, value in zip(model.metrics_names, test_results):
    print(f"{name}: {value:.4f}")

test_prob = model.predict(X_test_pca).ravel()
test_pred = (test_prob >= best_threshold).astype(int)

test_auc = roc_auc_score(y_test, test_prob)
print(f"\nTest ROC-AUC (sklearn): {test_auc:.4f}")

print("\n==============================")
print("Classification Report (TEST, trained with oversampling)")
print("(threshold = best F1 on validation)")
print("==============================")
print(classification_report(y_test, test_pred, digits=3))

cm_test = confusion_matrix(y_test, test_pred)
print("Confusion Matrix (TEST):")
print(cm_test)
plot_confusion(cm_test, "Confusion Matrix (TEST, trained with oversampling)")

fpr, tpr, _ = roc_curve(y_test, test_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {test_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve on Test Set")
plt.legend()
plt.show()


# ============================================================
# 9) Cross-validation with AUC only (kept same structure as yours)
# ============================================================
print("\n===== 5-Fold Stratified Cross-Validation (AUC only) =====")

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_aucs = []

fold_idx = 1
for train_idx, val_idx in kfold.split(X, y):
    print(f"\n--- Fold {fold_idx} ---")

    X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

    preprocessor_cv = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    X_tr_proc = preprocessor_cv.fit_transform(X_tr)
    X_va_proc = preprocessor_cv.transform(X_va)

    X_tr_dense = to_dense(X_tr_proc)
    X_va_dense = to_dense(X_va_proc)

    pca_cv = PCA(n_components=0.95, random_state=42)
    X_tr_pca_cv = pca_cv.fit_transform(X_tr_dense)
    X_va_pca_cv = pca_cv.transform(X_va_dense)

    classes_cv = np.unique(y_tr)
    cw = compute_class_weight(class_weight="balanced", classes=classes_cv, y=y_tr)
    cw_dict = {cls: w for cls, w in zip(classes_cv, cw)}

    model_cv = build_model(X_tr_pca_cv.shape[1])
    early_stop_cv = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        patience=5,
        mode="max",
        restore_best_weights=True,
        verbose=0
    )

    model_cv.fit(
        X_tr_pca_cv, y_tr,
        validation_data=(X_va_pca_cv, y_va),
        epochs=50,
        batch_size=64,
        class_weight=cw_dict,
        callbacks=[early_stop_cv],
        verbose=0
    )

    y_va_prob = model_cv.predict(X_va_pca_cv).ravel()
    fold_auc = roc_auc_score(y_va, y_va_prob)
    fold_aucs.append(fold_auc)
    print(f"Fold {fold_idx} ROC-AUC: {fold_auc:.4f}")

    fold_idx += 1

print("\nCross-validation AUCs:", [round(a, 4) for a in fold_aucs])
print("Mean CV AUC:", np.mean(fold_aucs).round(4))
print("Std CV AUC:", np.std(fold_aucs).round(4))
