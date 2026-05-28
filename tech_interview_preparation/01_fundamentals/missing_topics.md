# 📚 Missing Topics — MLOps, AutoML, Clustering, Mathematics

---

## Topic 1: MLOps & Production

### Flask Deployment
```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    X = np.array(data).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0, 1]
    return jsonify({'prediction': int(prediction), 'probability': float(probability)})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
```

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

```bash
# Build + Run
docker build -t ml-model .
docker run -p 5000:5000 ml-model

# Test
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.2, 3.4, 5.6]}'
```

### MLflow Tracking
```python
import mlflow
import mlflow.sklearn

mlflow.set_experiment("churn_prediction")

with mlflow.start_run():
    # Entrenar modelo
    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)
    
    # Log parámetros
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    
    # Log métricas
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log modelo
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Run ID: {mlflow.active_run().info.run_id}")
```

### Data Drift Detection
```python
from scipy.stats import ks_2samp

def detect_drift(X_train, X_today, threshold=0.05):
    drift_detected = []
    for col in X_train.columns:
        stat, p_value = ks_2samp(X_train[col], X_today[col])
        if p_value < threshold:
            drift_detected.append(col)
    return drift_detected

drift_cols = detect_drift(X_train, X_today)
if drift_cols:
    print(f"⚠️ Drift en: {drift_cols}")
```

---

## Topic 2: AutoML & Hyperparameter Tuning

### Grid Search
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.4f}")
```

### Random Search (más eficiente)
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_distributions=param_dist,
    n_iter=50,       # Solo 50 combinaciones vs todas en grid
    cv=5,
    scoring='f1',
    random_state=42
)
random_search.fit(X_train, y_train)
```

### Optuna (avanzado)
```python
import optuna

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    
    return cross_val_score(model, X_train, y_train, cv=3, scoring='f1').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best params: {study.best_params}")
print(f"Best F1: {study.best_value:.4f}")
```

---

## Topic 3: Clustering Avanzado

### PCA + Visualización
```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Reducir a 2D para visualizar
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

# Cluster
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Visualizar
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.title('KMeans Clusters (PCA 2D)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")
```

### Elbow Method + Silhouette
```python
from sklearn.metrics import silhouette_score

inertias = []
silhouettes = []
K_range = range(2, 10)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))

# Plotear
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(K_range, inertias, 'o-')
axes[0].set_title('Elbow Method')
axes[0].set_xlabel('K')
axes[0].set_ylabel('Inertia')

axes[1].plot(K_range, silhouettes, 'o-')
axes[1].set_title('Silhouette Score')
axes[1].set_xlabel('K')
axes[1].set_ylabel('Score')

plt.show()

best_k = K_range[np.argmax(silhouettes)]
print(f"Mejor K por Silhouette: {best_k}")
```

### t-SNE (visualización alta dimensión)
```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.title('t-SNE Visualization')
plt.show()
```

### Gaussian Mixture Models (overlapping clusters)
```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=4, random_state=42)
gmm.fit(X_scaled)

# Soft assignments
probs = gmm.predict_proba(X_scaled)  # shape: (n_samples, n_clusters)
labels_hard = gmm.predict(X_scaled)

print("Probabilidades (soft assignments):")
print(probs[:5].round(3))
```

---

## Topic 4: Mathematics Behind ML

### Linear Algebra (key concepts)
```
Dot product: a · b = Σ(aᵢ * bᵢ)
Matrix multiply: A @ B
Transpose: A.T
Inverse: np.linalg.inv(A)

Eigenvectors/Eigenvalues:
A * v = λ * v
PCA usa eigendecomposition de la covarianza
```

### Gradient Descent
```
θ = θ - α * ∇L(θ)

α = learning rate
∇L = gradiente del loss

Variants:
- Batch: todos los datos
- Stochastic: 1 muestra
- Mini-batch: batch de 32/64/128
```

### Probabilidad & Estadística
```
Bayes Theorem:
P(A|B) = P(B|A) * P(A) / P(B)

MLE (Maximum Likelihood Estimation):
Encontrar parámetros que maximizan P(data|params)

MAP (Maximum A Posteriori):
Encontrar parámetros que maximizan P(params|data) = P(data|params) * P(params)
```

### Métricas de Distancia
```python
from scipy.spatial.distance import euclidean, cosine

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Euclidean: distancia física
euclidean_dist = np.sqrt(np.sum((a - b)**2))

# Cosine: similitud de dirección (embeddings)
cosine_sim = 1 - cosine(a, b)

# Manhattan: suma absoluta
manhattan = np.sum(np.abs(a - b))
```

---

## Topic 5: Agile & Data Engineering (brevísimo)

### Agile/Scrum Keywords
```
Sprint: período fijo (2 semanas)
Backlog: lista de tareas pendientes
Daily standup: qué hice, qué haré, qué bloquea
Retrospective: qué mejorar al final de sprint
Velocity: puntos completados por sprint
```

### ETL Pipeline básico
```python
def etl_pipeline(source_path, dest_path):
    # Extract
    df = pd.read_csv(source_path)
    
    # Transform
    df = clean_pipeline(df)
    df = create_features(df)
    
    # Load
    df.to_parquet(dest_path, index=False)
    print(f"✅ ETL complete: {len(df)} rows → {dest_path}")
```

---

## 📝 Preguntas Adicionales (del JD)

**Q_MLOps.1:** "How would you deploy a model using Flask and Docker?"
→ Ver Topic 1: Flask + Dockerfile

**Q_MLOps.2:** "What is concept drift and how do you handle it?"
→ Ver Topic 1: Data Drift Detection + Q33 en general_questions.md

**Q_AutoML.1:** "Explain Grid Search vs Random Search vs Bayesian optimization."
→ Ver Topic 2: Grid Search vs Random Search vs Optuna

**Q_Math.1:** "Explain the mathematics behind gradient descent."
→ Ver Topic 4: Gradient Descent

**Q_Clustering.1:** "How do you choose the optimal number of clusters?"
→ Ver Topic 3: Elbow Method + Silhouette

