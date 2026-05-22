# Module 2: Data Processing — Preparación de Datos

## 🎯 Objetivos del módulo
- Manejar valores faltantes (missing values)
- Detectar y tratar outliers
- Scaling y normalización
- Encoding de variables categóricas
- Feature selection

**Tiempo:** 30 min lectura + 30 min ejercicios

---

## 2.1 Valores Faltantes (Missing Values)

### Entendiendo el problema

Los datos reales rara vez son perfectos. Datos faltantes (NaN, None, -999) son comunes.

```
Dataset:
Age  Income  Employed
25   50000   True
32   NaN     True      ← Falta Income
28   65000   False
NaN  55000   True      ← Falta Age
45   80000   NaN       ← Falta Employed
```

### Tipos de datos faltantes

**MCAR (Missing Completely At Random)**
- No hay relación con otras variables
- Proceso aleatorio puro
- Tratamiento: eliminar o imputar aleatoriamente

**MAR (Missing At Random)**
- Falta relacionada con otras variables, no con el valor en sí
- Ejemplo: personas de bajo ingreso menos probable reportar edad
- Tratamiento: imputación sofisticada

**MNAR (Missing Not At Random)**
- Falta relacionada con el valor que falta
- Ejemplo: personas con salarios muy altos omiten reportar
- Tratamiento: debe conocerse la razón, imputación con cuidado

### Estrategias de manejo

#### 1. Eliminar (Deletion)

```python
# Eliminar filas con NaN en cualquier columna
df_clean = df.dropna()

# Eliminar si falta en columna específica
df_clean = df.dropna(subset=['Age'])

# Eliminar si falta en MÁS del 50% de datos
df_clean = df.dropna(thresh=len(df)*0.5)
```

**Cuándo usar:**
- Pocos missing values (<5%)
- Datos MCAR
- Dataset grande

**Desventajas:**
- Pierdes información
- Sesgo si no es MCAR

#### 2. Imputación: Media/Mediana/Moda

```python
from sklearn.impute import SimpleImputer

# Media
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Mediana (más robusto a outliers)
imputer = SimpleImputer(strategy='median')

# Moda (para categóricas)
imputer = SimpleImputer(strategy='most_frequent')
```

**Cuándo usar:**
- Variables numéricas con distribución normal
- Missing <20%

**Desventajas:**
- Ignora relaciones entre variables
- Reduce varianza (subestima incertidumbre)

#### 3. Imputación: Forward/Backward Fill (Series de tiempo)

```python
# Para datos temporales
df['Age'] = df['Age'].fillna(method='ffill')  # Forward fill
df['Age'] = df['Age'].fillna(method='bfill')  # Backward fill
```

#### 4. Imputación: KNN

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
```

**Cuándo usar:**
- Relaciones complejas entre variables
- Missing MAR/MNAR

#### 5. Imputación: MICE (Multiple Imputation by Chained Equations)

```python
from sklearn_pandas import DataFrameMapper
# O usar: from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer()
X_imputed = imputer.fit_transform(X)
```

### Decision Tree: ¿Qué hacer?

```
¿Cuántos faltantes?
├─ <5% → Eliminar (delete)
├─ 5-20% → Imputación simple (media/mediana)
├─ >20% → Imputación avanzada (KNN, MICE)
│         O crear feature "missing indicator"
│
¿Tipo de falta?
├─ MCAR → Media/mediana ok
├─ MAR/MNAR → KNN/MICE mejor
```

### Crear feature "missing indicator"

A veces, el hecho de que falte es información:

```python
# Para cada columna con NaN, crea feature booleana
df['Age_missing'] = df['Age'].isna().astype(int)
```

---

## 2.2 Outliers (Valores Atípicos)

### Detectar outliers

#### Método 1: Rango Intercuartílico (IQR)

```python
Q1 = df['Income'].quantile(0.25)
Q3 = df['Income'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Income'] < lower_bound) | (df['Income'] > upper_bound)]
```

#### Método 2: Z-score

```python
from scipy import stats

z_scores = stats.zscore(df['Income'])
outliers = df[np.abs(z_scores) > 3]  # 3 desv. estándar
```

#### Método 3: Isolation Forest

```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.05)
outlier_labels = iso_forest.fit_predict(X)
outliers = X[outlier_labels == -1]
```

### Tratar outliers

#### Opción 1: Eliminar

```python
df_clean = df[(df['Income'] > lower_bound) & (df['Income'] < upper_bound)]
```

**Cuándo:** Errores de medición, datos corruptos

#### Opción 2: Capping (Winsorization)

```python
df['Income_capped'] = df['Income'].clip(lower=lower_bound, upper=upper_bound)
```

**Cuándo:** Outliers legítimos pero extremos

#### Opción 3: Transformación

```python
# Log: para distribuciones sesgadas
df['Income_log'] = np.log1p(df['Income'])

# Box-Cox: generalización de log
from scipy.stats import boxcox
df['Income_boxcox'] = boxcox(df['Income'] + 1)[0]
```

**Cuándo:** Outliers extremos pero válidos

---

## 2.3 Scaling y Normalización

### ¿Por qué escalar?

Algunos algoritmos son sensibles a la escala de features:

```
Salario: 20,000 - 200,000
Edad: 18 - 65

Sin escalar: Salario domina (rango más grande)
Con escalar: Ambos en escala 0-1 o -1 a 1
```

**Algoritmos que necesitan scaling:**
- KNN (distancia)
- K-Means (distancia)
- SVM (margen)
- Regresiónlineal con regularización (penaliza weights grandes)
- Redes neuronales (convergencia)

**Algoritmos que NO:**
- Árboles de decisión
- Random Forest
- Gradient boosting

### Standardization (Z-score normalization)

```
X_scaled = (X - μ) / σ

Resultado: Media 0, Desviación estándar 1
Rango típico: -3 a +3
```

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # ¡Usa fit de train!
```

### Normalization (Min-Max)

```
X_normalized = (X - min) / (max - min)

Resultado: Rango 0 a 1
```

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Cuándo usar cada una

| Método | Rango | Sensible a outliers | Cuándo |
|--------|-------|-------------------|--------|
| StandardScaler | -3 a +3 | No | Normal distribution, general |
| MinMaxScaler | 0 a 1 | Sí | Datos limitados, redes neuronales |
| RobustScaler | No acotado | No | Muchos outliers |

---

## 2.4 Encoding de Variables Categóricas

### El problema

Los algoritmos usan números. Variables categóricas (color, ciudad, género) son strings.

```python
Color: ['Red', 'Blue', 'Green', 'Red']
# ¿Cómo lo representa el modelo?
```

### One-Hot Encoding

Crea columna binaria por categoría:

```
Color original: ['Red', 'Blue', 'Green', 'Red']

One-hot:
    Red  Blue  Green
0   1    0     0
1   0    1     0
2   0    0     1
3   1    0     0
```

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X[['Color']])
```

**Cuándo usar:**
- Número bajo de categorías (<10)
- Relación no-ordinal

**Desventajas:**
- Muchos features si muchas categorías (curse of dimensionality)
- Multicolinealidad (suma = 1)

### Label Encoding

Asigna número a cada categoría:

```
Color: ['Red', 'Blue', 'Green', 'Red']
       [0,    1,      2,       0]
```

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
X['Color_encoded'] = encoder.fit_transform(X['Color'])
```

**Cuándo usar:**
- Árboles de decisión
- Muchas categorías (>10)
- Relación ordinal (malo < neutral < bueno)

**Desventajas:**
- Implica orden donde puede no haber (red=0, blue=1, verde=2 ¿por qué?)

### Target Encoding (Mean Encoding)

Reemplaza categoría por promedio del target:

```
Categoria  Promedio(target)
Red        0.72
Blue       0.55
Green      0.61

Color: ['Red', 'Blue', 'Green', 'Red']
       [0.72, 0.55,  0.61,   0.72]
```

```python
target_encoded = df.groupby('Color')['Target'].mean()
df['Color_encoded'] = df['Color'].map(target_encoded)
```

**Cuándo usar:**
- Muchas categorías
- Relación fuerte con target
- Modelos con regularización

**Cuidado:** ⚠️ Data leakage si no usas cross-validation

---

## 2.5 Feature Selection

### ¿Por qué seleccionar features?

- Menos features = modelo más simple
- Menos ruido = mejor generalización
- Entrenar más rápido
- Interpretabilidad

### Métodos

#### 1. Varianza

Elimina features con varianza baja (casi constantes):

```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)
```

#### 2. Correlación con target

Mantén features correlacionados con target:

```python
# Correlación de Pearson
correlation = df.corr()['Target'].sort_values(ascending=False)
selected_features = correlation[correlation > 0.3].index.tolist()
```

#### 3. Importancia del modelo

Usa feature importance de árbol o permutation:

```python
# Basado en gini/gain
model.fit(X_train, y_train)
importances = model.feature_importances_
important_features = X_train.columns[importances > 0.01]
```

#### 4. RFE (Recursive Feature Elimination)

Elimina features iterativamente:

```python
from sklearn.feature_selection import RFE

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=5)
X_selected = rfe.fit_transform(X_train, y_train)
```

#### 5. L1 Regularization (Lasso)

Automáticamente zeroa features irrelevantes:

```python
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
selected = X_train.columns[lasso.coef_ != 0]
```

---

## 2.6 Pipeline: Todo junto

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif, k=5)),
    ('model', LogisticRegression())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

**Ventaja:** Train/test split automático ✓ (no data leakage)

---

## 🎓 Conceptos Clave Resumidos

| Problema | Síntoma | Solución |
|----------|---------|----------|
| Missing values | NaN en dataset | Eliminar o imputar (media, KNN, MICE) |
| Outliers | Valores extremos | Detectar (IQR, Z-score), eliminar o cap |
| Escalas diferentes | Feature A >> Feature B | StandardScaler o MinMaxScaler |
| Categóricas | Strings en dataset | One-hot (pocos) o Label (muchos) |
| Muchos features | Ruido, lentitud | Feature selection (correlación, RFE, L1) |

---

## 🛠️ Checklist Data Processing

```
☐ Datos cargados
☐ Revisar missing values (porcentaje, patrón)
☐ Decidir estrategia: eliminar/imputar
☐ Detectar outliers (IQR, Z-score)
☐ Decidir: eliminar/cap/transformar
☐ Codificar categóricas
☐ Escalar (si es necesario)
☐ Feature selection (si muchos features)
☐ Validar en train/val/test
☐ Listo para modelar
```

---

## 📚 Lecturas Complementarias

- Pandas documentation: fillna, dropna
- Scikit-learn: preprocessing, impute
- "Feature Engineering for Machine Learning" — Alice Zheng

---

**Next:** Module 3 — Model Development
