# Module 3: Model Development — Desarrollo de Modelos

## 🎯 Objetivos del módulo
- Entender modelos clásicos (Logistic Regression, Decision Trees, Random Forest, XGBoost)
- Saber cuándo usar cada uno
- Entrenar y ajustar hiperparámetros
- Implementar A/B testing y quasi-experiments

**Tiempo:** 30 min lectura + 30 min ejercicios

---

## 3.1 Landscape de Modelos

### Tipos de problemas

```
┌─ Supervised Learning
│  ├─ Regression (predecir número continuo)
│  │  └─ Ejemplos: precio casa, temperatura, ingresos
│  └─ Classification (predecir categoría)
│     ├─ Binary (2 clases)
│     │  └─ Ejemplo: fraude sí/no, compra sí/no
│     └─ Multi-class (>2 clases)
│        └─ Ejemplo: iris (setosa/versicolor/virginica)
│
└─ Unsupervised Learning
   ├─ Clustering (agrupar datos similares)
   ├─ Dimensionality Reduction (reducir features)
   └─ Anomaly Detection
```

---

## 3.2 Logistic Regression

### ¿Qué es?

A pesar del nombre, es **clasificación**, no regresión. Predice probabilidad de clase positiva.

```
Entrada: X (features)
         ↓
    θ₀ + θ₁X₁ + θ₂X₂ + ...  (combinación lineal)
         ↓
    Sigmoid (convierte a probabilidad 0-1)
         ↓
Salida: P(Y=1|X)
```

### Función Sigmoid

```
σ(z) = 1 / (1 + e^(-z))

Gráfico:
    P(Y=1)
      |    ____
      |  _/
      | /
      |____________
         z →
```

**Propiedades:**
- z = -∞ → σ ≈ 0
- z = 0 → σ = 0.5
- z = +∞ → σ ≈ 1

### Entrenamiento

Minimiza **log-loss** (binary cross-entropy):

```
Loss = -[y*log(ŷ) + (1-y)*log(1-ŷ)]

Intuición:
- Si y=1 y ŷ≈1: Loss ≈ 0 ✓
- Si y=1 y ŷ≈0: Loss → ∞ ✗
- Si y=0 y ŷ≈0: Loss ≈ 0 ✓
- Si y=0 y ŷ≈1: Loss → ∞ ✗
```

### Ventajas y desventajas

| Aspecto | Evaluación |
|--------|-----------|
| Interpretabilidad | ⭐⭐⭐⭐⭐ (puedo explicar coefs) |
| Velocidad | ⭐⭐⭐⭐⭐ (muy rápido) |
| Accuracy | ⭐⭐⭐ (ok, no el mejor) |
| Relaciones no-lineales | ⭐ (asume linealidad) |
| Scaling required | Sí |

### Cuándo usar

- Baseline (compara contra esto)
- Explicabilidad crítica (crédito, medicina)
- Datos linealmente separables
- Entrenar rápido

### Implementación

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train, y_train)

# Probabilidades
y_pred_proba = model.predict_proba(X_test)  # [[P(0), P(1)], ...]

# Clases
y_pred = model.predict(X_test)  # [0, 1, 1, 0, ...]

# Coeficientes (interpretables)
for feature, coef in zip(X_train.columns, model.coef_[0]):
    print(f"{feature}: {coef:.4f}")
```

---

## 3.3 Decision Trees

### ¿Qué es?

Árbol de decisiones binarias que divide el espacio de features.

```
¿Edad < 30?
├─ Sí: ¿Ingreso < 50k?
│      ├─ Sí: Clase A
│      └─ No: Clase B
└─ No: ¿Años experiencia > 5?
       ├─ Sí: Clase B
       └─ No: Clase A
```

### Cómo crece el árbol

En cada nodo, elige el feature que mejor **separa** las clases:

**Métrica: Gini o Entropy**

```
Gini = 1 - Σ(p_i²)

Ejemplo:
Nodo puro (100% clase A): Gini = 1 - 1² = 0 ✓
Nodo mixed 50/50: Gini = 1 - 0.5² - 0.5² = 0.5
```

Elige split que **minimiza Gini ponderado**.

### Ventajas y desventajas

| Aspecto | Evaluación |
|--------|-----------|
| Interpretabilidad | ⭐⭐⭐⭐⭐ (visualizable) |
| No requiere scaling | ⭐⭐⭐⭐⭐ (invariante) |
| Accuracy | ⭐⭐⭐ (ok) |
| Relaciones no-lineales | ⭐⭐⭐⭐ (sí) |
| Overfitting | ⭐ (tiende a overfittear) |

### Hiperparámetros importantes

```python
tree = DecisionTreeClassifier(
    max_depth=5,              # Profundidad máxima
    min_samples_split=10,     # Mínimo de muestras para split
    min_samples_leaf=5,       # Mínimo en cada hoja
    criterion='gini'          # O 'entropy'
)
```

- **max_depth bajo** → underfitting
- **max_depth alto** → overfitting

### Cuándo usar

- Datos con relaciones no-lineales
- Interpretabilidad importante
- Bajar costo de baseline

---

## 3.4 Random Forest

### ¿Qué es?

Ensemble de árboles independientes que votean. Cada árbol es profundo (no limitado).

```
Dataset original
    ↓
Bootstrap sample 1 → Árbol 1
Bootstrap sample 2 → Árbol 2
Bootstrap sample 3 → Árbol 3
... 100 árboles
    ↓
Predicción = PROMEDIO (regresión) o VOTO MAYORITARIO (clasificación)
```

### Ventajas del ensemble

```
Árbol individual:
├─ Overfittea fácil
└─ Variance alta

Random Forest:
├─ Promedio reduce variance
├─ Viés bajo (árboles profundos)
└─ Resultado: bajo sesgo + baja varianza = MEJOR
```

### Paralelización

Cada árbol es **independiente** → entrenar en paralelo = MÁS RÁPIDO

### Ventajas y desventajas

| Aspecto | Evaluación |
|--------|-----------|
| Accuracy | ⭐⭐⭐⭐ (muy bueno) |
| Interpretabilidad | ⭐⭐ (black-box) |
| Relaciones no-lineales | ⭐⭐⭐⭐⭐ |
| Speed | ⭐⭐⭐⭐ (rápido) |
| Overfitting | ⭐⭐⭐⭐ (resistente) |
| Scaling required | No |

### Hiperparámetros

```python
rf = RandomForestClassifier(
    n_estimators=100,         # Número de árboles
    max_depth=10,             # Profundidad individual
    min_samples_split=5,
    n_jobs=-1                 # Paralelo: -1 = todos cores
)
```

### Cuándo usar

- Cuando accuracy es prioridad
- Datos complejos, no-lineales
- Tienes suficientes datos
- Rápido baseline

### Feature Importance

```python
rf.fit(X_train, y_train)
importances = rf.feature_importances_

# Visualizar
import matplotlib.pyplot as plt
plt.barh(X_train.columns, importances)
plt.show()
```

---

## 3.5 XGBoost (Extreme Gradient Boosting)

### ¿Qué es?

Boosting secuencial: cada árbol intenta **corregir** el error del anterior.

```
Dataset
  ↓
Árbol 1 predice → Error 1
  ↓
Árbol 2 entrena en Residuales 1 → Error 2
  ↓
Árbol 3 entrena en Residuales 2 → Error 3
...
  ↓
Predicción = Sum(Árbol 1 + Árbol 2 + ...)
```

### Ventajas sobre Random Forest

```
Random Forest:
  └─ Árboles independientes, votean

XGBoost:
  └─ Árboles colaborativos, cooperan
     (siguiente corrige el anterior)
```

Resultado: **XGBoost típicamente > Random Forest en accuracy**

### Ventajas y desventajas

| Aspecto | Evaluación |
|--------|-----------|
| Accuracy | ⭐⭐⭐⭐⭐ (mejor) |
| Interpretabilidad | ⭐⭐ (black-box) |
| Speed | ⭐⭐ (lento, secuencial) |
| Regularización incorporada | ⭐⭐⭐⭐⭐ (built-in) |
| Overfitting resistance | ⭐⭐⭐⭐ |
| Scaling required | No |

### Hiperparámetros importantes

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=100,       # Árboles
    max_depth=5,            # Profundidad (XGB: bajo es mejor)
    learning_rate=0.1,      # Tasa de aprendizaje (eta)
    subsample=0.8,          # Muestra por árbol (evita overfitting)
    colsample_bytree=0.8,   # Features por árbol
    reg_lambda=1.0,         # L2 regularización
    reg_alpha=0.0           # L1 regularización
)
```

- **learning_rate bajo** + **n_estimators alto** = mejor generalización

### Cuándo usar

- Competencias (Kaggle)
- Máxima accuracy deseada
- Datos complejos
- Tienes CPU decente (es computacionalmente intensivo)

---

## 3.6 Comparación Rápida

| Modelo | Accuracy | Speed | Interpretable | Escalar | Cuándo |
|--------|----------|-------|---------------|---------|--------|
| Logistic Reg | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Sí | Baseline |
| Decision Tree | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | No | Simple |
| Random Forest | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | No | Good balance |
| XGBoost | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | No | Max accuracy |

---

## 3.7 Estrategia de modelado

### Paso 1: Baseline

Empieza simple:

```python
# Dummy classifier
from sklearn.dummy import DummyClassifier
baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train, y_train)
print(f"Baseline accuracy: {baseline.score(X_test, y_test)}")
```

**Todos los modelos deben vencer baseline.**

### Paso 2: Logistic Regression

Segundo modelo simple:

```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
print(f"LR accuracy: {lr.score(X_test, y_test)}")
```

**¿Mejor que baseline? → Señal de que hay patrón**

### Paso 3: Ensemble (RF o XGB)

Si LR es prometedor:

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
print(f"RF accuracy: {rf.score(X_test, y_test)}")
```

### Paso 4: Tuning

Optimiza hiperparámetros del mejor:

```python
from sklearn.model_selection import GridSearchCV

params = {
    'max_depth': [5, 10, 15],
    'n_estimators': [50, 100, 200]
}

grid = GridSearchCV(RandomForestClassifier(), params, cv=5)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_}")
```

---

## 3.8 A/B Testing y Quasi-Experiments

### A/B Test: Comparar dos versiones

```
Grupo A (Control): Versión anterior
Grupo B (Treatment): Versión nueva

¿Hay diferencia significativa en métrica?
```

### Implementación

```python
from scipy import stats

# Dividir usuarios aleatoriamente
control = users[users['group'] == 'A']['metric']
treatment = users[users['group'] == 'B']['metric']

# T-test
t_stat, p_value = stats.ttest_ind(control, treatment)

if p_value < 0.05:
    print("Diferencia estadísticamente significativa")
else:
    print("Sin diferencia significativa")
```

### Quasi-experiments: Cuando no puedes hacer A/B

Usa métodos estadísticos:
- **Difference-in-Differences (DiD):** Compara cambio en tiempo
- **Regression Discontinuity (RDD):** Explota límites naturales

---

## 🎓 Conceptos Clave Resumidos

```
Baseline < Logistic Reg < Random Forest < XGBoost
```

Pero:
```
Interpretabilidad: Logistic Reg > Decision Tree > RF > XGBoost
```

Elige balance según contexto.

---

## 🛠️ Checklist Modelado

```
☐ Baseline (dummy)
☐ Logistic Regression
☐ Random Forest
☐ XGBoost
☐ Tuning (GridSearch)
☐ Cross-validation (5-fold)
☐ Comparar en test
☐ Feature importance
☐ Decisión: ¿Cuál a producción?
```

---

## 📚 Lecturas Complementarias

- "Introduction to Statistical Learning" — James et al.
- XGBoost docs: https://xgboost.readthedocs.io/
- Scikit-learn: model selection, ensemble

---

**Next:** Module 4 — Evaluation Metrics
