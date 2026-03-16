from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score


# ===============================
# 1. Wczytanie danych
# ===============================
wine = load_wine()
X = wine.data
y = wine.target

print("Liczba próbek:", X.shape[0])
print("Liczba cech:", X.shape[1])
print("Klasy:", wine.target_names)

# ===============================
# 2. Podział na train/test
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

# ===============================
# 3. Pipeline (skalowanie + KNN)
# ===============================
pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ])

# ===============================
# 4. Trenowanie modelu
# ===============================
pipeline.fit(X_train, y_train)

# ===============================
# 5. Predykcja
# ===============================
y_pred = pipeline.predict(X_test)

# ===============================
# 6. Ewaluacja
# ===============================
print("Accuracy (test set):", accuracy_score(y_test, y_pred))
print("\nMacierz pomyłek:\n", confusion_matrix(y_test, y_pred))
