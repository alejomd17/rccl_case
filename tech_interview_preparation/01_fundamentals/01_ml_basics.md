# Module 1: ML Basics — Fundamentos de Machine Learning

## 🎯 Objetivos del módulo
- Entender overfitting vs underfitting
- Dominar el bias-variance tradeoff
- Diferenciar train/validation/test split
- Aplicar cross-validation correctamente
- Conocer regularización (L1, L2)

**Tiempo:** 30 min lectura + 30 min ejercicios

---

## 1.1 Overfitting vs Underfitting

### ¿Qué es overfitting?

Un modelo **overfittea** cuando memoriza los datos de entrenamiento en lugar de aprender patrones generalizables.

**Síntomas:**
- Training loss: muy bajo ✓
- Validation loss: muy alto ✗
- Training accuracy: 98%
- Validation accuracy: 72%

**Por qué ocurre:**
- Modelo demasiado complejo (muchos parámetros)
- Datos insuficientes
- Entrenamiento excesivo (demasiadas épocas)

### ¿Qué es underfitting?

Un modelo **underfittea** cuando es demasiado simple para capturar los patrones.

**Síntomas:**
- Training loss: alto
- Validation loss: alto (similar a training)
- Training accuracy: 65%
- Validation accuracy: 68%

**Por qué ocurre:**
- Modelo muy simple (pocos parámetros)
- Características pobres (feature engineering deficiente)
- Entrenamiento insuficiente

### Visual

```
Error

      Underfitting  |  Goldilocks  |  Overfitting
                    |              |
                    |    ___       |
      ___           |   /   \      |  /
     /   \          |  /     \___  | /
    /     \___      | /           \|/____
   /            \___|/              \
  __________________|________________
  Complejidad del modelo →
```

---

## 1.2 Bias-Variance Tradeoff

### Desglose del error

Todo error de predicción se compone de tres elementos:

```
Error Total = Bias² + Variance + Ruido irreducible
```

### Bias (sesgo)

- **Definición:** Error promedio entre predicciones esperadas y valor real
- **Causa:** Modelo demasiado simple
- **Síntoma:** Underfitting
- **Ejemplo:** Usar regresión lineal para datos no-lineales

**Bias alto:**
```
Datos:     xxxxxx
           xx  xx
Predicción: _______ (línea plana — muy simple)
```

### Variance (varianza)

- **Definición:** Variabilidad de predicciones entre diferentes entrenamientos
- **Causa:** Modelo demasiado complejo / sensible a ruido
- **Síntoma:** Overfitting
- **Ejemplo:** Polinomio grado 10 para datos sencillos

**Variance alta:**
```
Datos:     xxxxxx
           xx  xx
Predicción: /\/\/\ (zig-zag — sigue cada punto)
```

### El tradeoff

- **Aumentar complejidad:** ↓ Bias, ↑ Variance
- **Disminuir complejidad:** ↑ Bias, ↓ Variance
- **Objetivo:** Encontrar el punto óptimo

```
Error

  |        Total Error
  |      /  \
  |     /    \
  |    /      \___
  | __/Variance   \____
  |/                   \___
  |                        \____
  |_____________________________
           Complejidad →
     (menos parámetros → más parámetros)
```

---

## 1.3 Train / Validation / Test Split

### Por qué tres conjuntos?

#### Training Set (60-70%)
- **Propósito:** Entrenar el modelo
- **Qué pasa:** Ajusta los pesos/parámetros
- **Métrica:** Loss que minimizas (no es honesta)

#### Validation Set (10-20%)
- **Propósito:** Tuning de hiperparámetros
- **Qué pasa:** Pruebas diferentes configuraciones (lr, regularización, etc.)
- **Métrica:** Evaluación intermedia
- **¿Por qué no test?** Si usas test para tuning, lo contaminas

#### Test Set (10-20%)
- **Propósito:** Evaluación final HONESTA
- **Qué pasa:** Nunca lo tocas durante entrenamiento
- **Métrica:** La métrica "verdadera" de producción
- **Regla de oro:** Solo miras 1 vez, al final

### Ejemplo gráfico

```
Dataset completo (1000 muestras)
|______________|______________|______________|
Train          Validation      Test
(700)          (150)           (150)
   ↓               ↓              ↓
Entrenar      Ajustar        Evaluar
parámetros    hiperparams    finalmente
```

### ¿Qué pasa si haces split mal?

```
❌ MAL: Escalar ANTES de split
X_scaled = scaler.fit_transform(X)  # Info de TODA la data
X_train, X_test = train_test_split(X_scaled)
→ Data leakage: test information filtra a train

✅ BIEN: Escalar DESPUÉS de split
X_train, X_test = train_test_split(X)
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Usa fit de train
→ Cada set vive en su burbuja
```

---

## 1.4 Cross-Validation

### El problema

Con train/val/test solo 1 vez, el resultado depende de qué muestras cayeron donde.

```
Split 1:
Train: [1,2,3,4,5,6,7] → Acc: 85%
Test: [8,9,10]

Split 2:
Train: [1,2,3,8,9,10] → Acc: 82%
Test: [4,5,6,7]
```

¿Cuál es el verdadero accuracy? **Ambos splits son válidos.**

### K-Fold Cross-Validation

Divide los datos en **k pliegues** iguales. Entrena k veces:
- Fold 1: entrenar con 2-k, validar con 1
- Fold 2: entrenar con 1,3-k, validar con 2
- ...
- Fold k: entrenar con 1-k-1, validar con k

```
Dataset: [1,2,3,4,5,6,7,8,9,10]

Fold 1: Train [2-10], Val [1]       → Acc: 85%
Fold 2: Train [1,3-10], Val [2]     → Acc: 83%
Fold 3: Train [1,2,4-10], Val [3]   → Acc: 87%
...
Fold 5: Train [1-9], Val [10]       → Acc: 84%

Promedio: 84.8% ± 1.5%  ← Más robusto
```

### Variantes

**Stratified K-Fold:**
Para clasificación con clases desbalanceadas. Asegura que cada fold tenga la misma proporción de clases.

```
Dataset balanceado:
Class 0: 50%, Class 1: 50%

Cada fold:
Class 0: 50%, Class 1: 50%  ← Sin variación
```

**Leave-One-Out CV:**
k = n (cada muestra es un fold).
- Más costoso computacionalmente
- Útil con datasets muy pequeños

---

## 1.5 Regularización (L1 y L2)

### El problema

Un modelo con muchos parámetros tiende a overfittear. Solución: penalizar parámetros grandes.

### L2 Regularization (Ridge)

Agrega una **penalidad proporcional al cuadrado** de los pesos.

```
Loss = MSE + λ * Σ(w²)
                 todos los pesos

λ (lambda) = fuerza de regularización (hiperparámetro)
```

**Efecto:**
- Mantiene todos los pesos pequeños
- Penaliza todos los features por igual
- Weights → cercanos a cero, nunca exactamente cero

**Cuándo usar:**
- Muchos features relevantes
- Quieres evitar overfitting pero mantener todas las features

### L1 Regularization (Lasso)

Agrega una **penalidad proporcional al valor absoluto** de los pesos.

```
Loss = MSE + λ * Σ(|w|)
                 todos los pesos
```

**Efecto:**
- Algunos pesos → exactamente cero
- **Feature selection automática**
- Solo features importantes sobreviven

**Cuándo usar:**
- Muchos features, sospechas que muchos son irrelevantes
- Quieres interpretabilidad (qué features importan)

### Elastic Net

Combina L1 y L2:

```
Loss = MSE + λ₁ * Σ(|w|) + λ₂ * Σ(w²)
```

---

## 1.6 Regularización en la Práctica

### Ejemplo: Regresión

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# Sin regularización
lr = LinearRegression()
lr.fit(X_train, y_train)

# Con L2 (Ridge)
ridge = Ridge(alpha=1.0)  # alpha = λ
ridge.fit(X_train, y_train)

# Con L1 (Lasso)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print(f"Coeficientes cero: {(lasso.coef_ == 0).sum()}")

# Elastic Net
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet.fit(X_train, y_train)
```

### Cómo elegir λ (alpha)

- **λ muy bajo:** Poco efecto, overfitting
- **λ muy alto:** Todos los pesos → 0, underfitting
- **Solución:** Cross-validation

```python
from sklearn.linear_model import RidgeCV

ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
ridge_cv.fit(X_train, y_train)
print(f"Mejor alpha: {ridge_cv.alpha_}")
```

---

## 🎓 Conceptos Clave Resumidos

| Concepto | Qué es | Síntoma | Solución |
|----------|--------|---------|----------|
| **Overfitting** | Memoriza datos | Val loss >> train loss | Regularización, más datos |
| **Underfitting** | Modelo muy simple | Train loss alto | Modelo más complejo, features |
| **Bias alto** | Error sistemático | Predicciones siempre lejos | Modelo más complejo |
| **Variance alta** | Sensible a ruido | Varía mucho entre folds | Regularización, datos |
| **Data leakage** | Info del futuro filtra | Test >> val performance | Pipeline limpio |

---

## 🛠️ Quick Reference: Debugging

```
Problema: Modelo malo en general
├─ ¿Training accuracy bajo?
│  └─ Underfitting: más parámetros, features, épocas
│
└─ ¿Training accuracy alto, validation baja?
   └─ Overfitting: regularización, early stopping, menos features

Problema: Entrenamiento lento
├─ ¿Muchas muestras?
│  └─ SGD en lugar de GD
│
└─ ¿Muchos features?
   └─ Feature selection, regularización L1

Problema: Métrica inconsistente
└─ Cross-validation: verifica en múltiples folds
```

---

## 📚 Lecturas Complementarias

- StatQuest: Bias-Variance Tradeoff (YouTube)
- Hands-On ML: Capítulos 1-4
- Scikit-learn docs: Cross-validation

---

**Next:** Module 2 — Data Processing
