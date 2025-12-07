import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras import layers, models

np.set_printoptions(precision=3, suppress=True)


# 1. Load dataset
data_path = "online_shoppers_intention.csv"   # make sure this matches your file name
df = pd.read_csv(data_path)

print("Shape of dataset:", df.shape)
print("\nColumns:", df.columns)
print("\nFirst 5 rows:")
print(df.head())


# 2. Target distribution (Revenue)
print("\nRevenue value counts:")
print(df["Revenue"].value_counts())
print("\nRevenue proportion:")
print(df["Revenue"].value_counts(normalize=True))

plt.figure()
df["Revenue"].value_counts().plot(kind="bar")
plt.title("Class Distribution of Revenue")
plt.xticks(rotation=0)
plt.xlabel("Revenue")
plt.ylabel("Count")
plt.show()


# 3. Features (X) and target (y)
X = df.drop("Revenue", axis=1)
y = df["Revenue"].astype(int)   # True/False -> 1/0

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)


# 4. Train / validation / test split
#    70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,      # 15% of total
    random_state=42,
    stratify=y_temp
)

print("\nTrain size:", X_train.shape[0])
print("Val size:", X_val.shape[0])
print("Test size:", X_test.shape[0])


# 5. Preprocessing: scaling + one-hot encoding
categorical_cols = ["Month", "VisitorType", "Weekend"]
numeric_cols = [c for c in X.columns if c not in categorical_cols]

print("\nNumeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Fit on train only
X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc = preprocessor.transform(X_val)
X_test_proc = preprocessor.transform(X_test)

print("\nShape after preprocessing (train):", X_train_proc.shape)

# Convert to dense if sparse
def to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X

X_train_proc_dense = to_dense(X_train_proc)
X_val_proc_dense = to_dense(X_val_proc)
X_test_proc_dense = to_dense(X_test_proc)


# 6. Dimensionality reduction with PCA
#    Keep 95% of variance
pca = PCA(n_components=0.95, random_state=42)

X_train_pca = pca.fit_transform(X_train_proc_dense)
X_val_pca = pca.transform(X_val_proc_dense)
X_test_pca = pca.transform(X_test_proc_dense)

print("\nOriginal dim:", X_train_proc_dense.shape[1])
print("Reduced dim (PCA):", X_train_pca.shape[1])

explained = pca.explained_variance_ratio_
print("PCA explained variance (first 10 components):", explained[:10].round(3))
print("Total explained variance:", explained.sum().round(3))

input_dim = X_train_pca.shape[1]


# 7. Class weights for imbalance
classes = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weight_dict = {cls: w for cls, w in zip(classes, class_weights)}
print("\nClass weights:", class_weight_dict)


# 8. Build ANN with AUC + Accuracy
def build_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")  # binary classification
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),                   # main metric
            tf.keras.metrics.BinaryAccuracy(name="accuracy")    # secondary metric
        ]
    )
    return model

model = build_model(input_dim)
model.summary()


# 9. Train with early stopping (using val AUC)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc",
    patience=5,
    mode="max",
    restore_best_weights=True
)

history = model.fit(
    X_train_pca,
    y_train,
    validation_data=(X_val_pca, y_val),
    epochs=50,
    batch_size=64,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1
)

history_dict = history.history


# 10. Plot training vs validation AUC, loss, accuracy
# AUC
plt.figure()
plt.plot(history_dict["auc"], label="Train AUC")
plt.plot(history_dict["val_auc"], label="Val AUC")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("Training vs Validation AUC")
plt.legend()
plt.show()

# Loss
plt.figure()
plt.plot(history_dict["loss"], label="Train loss")
plt.plot(history_dict["val_loss"], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()

# Accuracy 
if "accuracy" in history_dict and "val_accuracy" in history_dict:
    plt.figure()
    plt.plot(history_dict["accuracy"], label="Train accuracy")
    plt.plot(history_dict["val_accuracy"], label="Val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.show()


# 11. Evaluate on test set with ROC-AUC & Accuracy
# Evaluate using Keras metrics first
test_results = model.evaluate(X_test_pca, y_test, verbose=0)
print("\nTest results (Keras):")
for name, value in zip(model.metrics_names, test_results):
    print(f"{name}: {value:.4f}")

# Get predicted probabilities
y_prob = model.predict(X_test_pca).ravel()
# Use 0.5 threshold by default for confusion matrix etc.
y_pred = (y_prob >= 0.5).astype(int)

# ROC-AUC using sklearn (more standard for reporting)
test_auc = roc_auc_score(y_test, y_prob)
print(f"\nTest ROC-AUC (sklearn): {test_auc:.4f}")

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {test_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve on Test Set")
plt.legend()
plt.show()

# Confusion matrix + classification report
print("\nClassification Report (threshold=0.5):")
print(classification_report(y_test, y_pred, digits=3))

print("Confusion Matrix (threshold=0.5):")
print(confusion_matrix(y_test, y_pred))


# 12. Cross-validation with AUC-ROC (5-fold)
print("\n===== 5-Fold Stratified Cross-Validation (AUC only) =====")

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_aucs = []

fold_idx = 1
for train_idx, val_idx in kfold.split(X, y):
    print(f"\n--- Fold {fold_idx} ---")

    X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

    # New preprocessor for each fold (to avoid data leakage)
    preprocessor_cv = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    X_tr_proc = preprocessor_cv.fit_transform(X_tr)
    X_va_proc = preprocessor_cv.transform(X_va)

    X_tr_dense = to_dense(X_tr_proc)
    X_va_dense = to_dense(X_va_proc)

    # PCA for each fold
    pca_cv = PCA(n_components=0.95, random_state=42)
    X_tr_pca = pca_cv.fit_transform(X_tr_dense)
    X_va_pca = pca_cv.transform(X_va_dense)

    # Class weights for this fold
    classes_cv = np.unique(y_tr)
    cw = compute_class_weight(
        class_weight="balanced",
        classes=classes_cv,
        y=y_tr
    )
    cw_dict = {cls: w for cls, w in zip(classes_cv, cw)}

    # Build and train model
    model_cv = build_model(X_tr_pca.shape[1])

    early_stop_cv = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        patience=5,
        mode="max",
        restore_best_weights=True,
        verbose=0
    )

    history_cv = model_cv.fit(
        X_tr_pca, y_tr,
        validation_data=(X_va_pca, y_va),
        epochs=50,
        batch_size=64,
        class_weight=cw_dict,
        callbacks=[early_stop_cv],
        verbose=0
    )

    # AUC on validation fold
    y_va_prob = model_cv.predict(X_va_pca).ravel()
    fold_auc = roc_auc_score(y_va, y_va_prob)
    fold_aucs.append(fold_auc)
    print(f"Fold {fold_idx} ROC-AUC: {fold_auc:.4f}")

    fold_idx += 1

print("\nCross-validation AUCs:", [round(a, 4) for a in fold_aucs])
print("Mean CV AUC:", np.mean(fold_aucs).round(4))
print("Std CV AUC:", np.std(fold_aucs).round(4))
