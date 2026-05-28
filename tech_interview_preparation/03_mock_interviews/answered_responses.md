# 📝 Answered Responses — 40 Mock Interview Questions

**Documento compilado:** Respuestas modelo para cada pregunta  
**Uso:** Compara tus respuestas con estas después de cada round  
**Nota:** Estas son respuestas esperadas; tu estilo puede variar

---

## ROUND 1: Warm-up Fundamentals (Jueves 21:00)

### Q1: Overfitting Basics

**Respuesta modelo:**

Overfitting ocurre cuando un modelo memoriza los datos de entrenamiento en lugar de aprender patrones generalizables. El modelo ajusta demasiado bien al ruido y peculiaridades del dataset de entrenamiento.

**Cómo detectarlo:**
- Training loss muy bajo vs validation loss muy alto (brecha grande)
- Training accuracy 98% vs validation accuracy 65% (diferencia significativa)
- Gráfico de learning curve: train converge, val diverge

**Cómo prevenirlo:**
- **Regularización:** L1/L2 penaliza pesos grandes
- **Early stopping:** detén entrenamiento cuando val loss empieza a subir
- **Más datos:** reduce el ruido relativo
- **Simplificar modelo:** menos parámetros, menos capas
- **Cross-validation:** valida en múltiples splits
- **Dropout:** en redes neuronales (apaga neuronas aleatoriamente)

**Ejemplo:** Ajustar polinomio grado 1 vs grado 10 a datos cuadráticos. Grado 10 overfittea (zig-zag).

---

### Q2: Train/Val/Test Split

**Respuesta modelo:**

Separamos en tres conjuntos porque cada uno tiene propósito diferente:

**Training Set (60-70%):**
- Entrenas el modelo
- Ajustas los pesos/parámetros
- La métrica aquí NO es honesta (el modelo ya vio estos datos)

**Validation Set (10-20%):**
- Ajustas hiperparámetros (learning rate, regularización, profundidad árbol)
- Pruebas diferentes configuraciones
- NO es para evaluación final

**Test Set (10-20%):**
- Evaluación FINAL honesta
- El modelo NUNCA vio estos datos
- Regla: miras solo 1 vez, al final
- Métrica aquí refleja verdadero rendimiento en producción

**¿Por qué no usar el mismo conjunto?**

Si usas el mismo para train y eval, inflás las métricas (el modelo ya memorizó esos datos). No sabes verdadero desempeño.

**Data leakage común:**
```python
❌ X_scaled = scaler.fit_transform(X)  # Info de toda data
   X_train, X_test = train_test_split(X_scaled)

✅ X_train, X_test = train_test_split(X)
   scaler.fit(X_train)
   X_train_scaled = scaler.transform(X_train)
   X_test_scaled = scaler.transform(X_test)
```

---

### Q3: Bias-Variance Tradeoff

**Respuesta modelo:**

Todo error de predicción se descompone en:
```
Error Total = Bias² + Variance + Ruido Irreducible
```

**Bias (sesgo):**
- Error promedio entre predicción y valor real
- Causa: modelo demasiado simple
- Síntoma: underfitting
- Ejemplo: usar línea recta para datos no-lineales
- Solución: modelo más complejo

**Variance (varianza):**
- Variabilidad de predicciones entre diferentes entrenamientos
- Causa: modelo demasiado complejo / sensible a ruido
- Síntoma: overfitting
- Ejemplo: polinomio grado 100 para datos sencillos
- Solución: regularización, más datos

**El tradeoff:**
- Aumentar complejidad: ↓ Bias, ↑ Variance
- Disminuir complejidad: ↑ Bias, ↓ Variance
- Objetivo: encontrar el punto óptimo en el medio

**Visualización mental:**
```
        Error
         |
         |     Total Error
         |    /  \
         |   /    \  Bias²
         |  /   ___\___
         | /___/       \___Variance
         |___________________
         Complejidad del modelo →
```

---

### Q4: Precision vs Recall (EPAM)

**Respuesta modelo:**

**Precision = TP / (TP + FP)**
- "De lo que predije positivo, cuánto acerté"
- Importa cuando falsas alarmas son costosas
- Ejemplos: spam filter, publicidad dirigida

**Recall = TP / (TP + FN)**
- "De lo positivo real, cuánto detecté"
- Importa cuando perder positivos es crítico
- Ejemplos: detección de cáncer, fraude

**Trade-off:**
Si subes threshold de predicción → más precisión, menos recall
Si bajas threshold → más recall, menos precisión

**Ejemplo: Fraude Bancario**
- Recall alto (99%): detecta casi todos los fraudes, pero muchas falsas alarmas
- Precision alta (99%): pocas falsas alarmas, pero algunos fraudes se escapan
- **Decisión:** Típicamente recall > precision (no queremos perder fraudes)

**Métricas combinadas:**
- F1 = 2 * (Precision * Recall) / (Precision + Recall) — balance
- ROC-AUC — trade-off visual

---

### Q5: Overfitting Prevention (EPAM)

**Respuesta modelo:**

**Definición:** Modelo memoriza datos en lugar de generalizar.

**Síntomas clave:**
- Training loss/error: muy bajo ✓
- Validation loss/error: muy alto ✗
- Gap grande entre train y validation

**Estrategias de prevención:**

1. **Regularización:**
   - L1/L2: penaliza pesos grandes
   - Elastic Net: combinación de ambas

2. **Early Stopping:**
   - Entrena hasta que validation loss empieza a subir
   - Detén ahí (no esperes convergencia total)

3. **Más datos:**
   - Reduce el peso relativo del ruido
   - Data augmentation (rotaciones, crops para imágenes)

4. **Simplificar modelo:**
   - Menos parámetros
   - Menos capas (redes neuronales)
   - max_depth más bajo (árboles)

5. **Cross-Validation:**
   - Valida en múltiples splits
   - Reduce varianza de una evaluación aleatoria

6. **Dropout (redes neuronales):**
   - Apaga neuronas aleatoriamente durante entrenamiento
   - Previene co-adaptación

7. **Batch Normalization:**
   - Normaliza activaciones entre capas
   - Mejora generalización

---

## ROUND 2: Model Selection & Data (Viernes 11:00-15:00)

### Q6: Precision vs Recall Context (Original)

**Respuesta modelo:**

**Contexto: Modelo detecta fraude bancario**

Optimizaría para **Recall** porque:
- Costo de falso negativo (fraude no detectado): muy alto 💰
- Costo de falso positivo (alerta falsa): menor (revisión manual)

**Métrica:** F1 considerando recall > precision

**Implementación:**
1. Ajusto threshold de probabilidad más bajo (más agresivo)
2. Monitoreo recall ≥ 95% como mínimo
3. Precision baja es acceptable si recall es alta

**Si fuera spam filter:**
- Optimizaría precision (no queremos emails válidos en spam)
- Recall bajo es acceptable

---

### Q7: Choosing Between Models

**Respuesta modelo:**

Con 100k registros, estrategia progresiva:

**1. Baseline:** Logistic Regression
- Rápido (minutos)
- Interpretable (coeficientes)
- Sirve de referencia

**2. Si LR es prometedor:** Random Forest
- Mejor que LR típicamente
- Velocidad aceptable (100k soporta bien)
- Parallelizable
- Feature importance automático

**3. Si RF no es suficiente:** XGBoost
- Mejor accuracy típicamente
- Requiere más tuning
- Más lento (pero 100k es manejable)

**Criterios de decisión:**

| Factor | Decisión |
|--------|----------|
| Interpretabilidad crítica | LR |
| Accuracy prioridad | XGBoost |
| Balance | Random Forest |
| Tiempo limitado | LR → RF |
| Feature eng necesario | XGBoost |

**Mi recomendación:** Start LR, compare en test. If LR.f1 > 0.75, keep it. Else try RF.

---

### Q8: Class Imbalance

**Respuesta modelo:**

**Problema:** 95% clase negativa, 5% positiva
- Accuracy es engañosa: modelo que predice siempre 0 = 95% accuracy
- No es útil

**Soluciones:**

1. **Class Weights:**
```python
model = LogisticRegression(class_weight='balanced')
```
Penaliza errores en clase minoritaria

2. **Oversampling (SMOTE):**
- Genera muestras sintéticas de clase minoritaria
- Aumenta tamaño de dataset

3. **Undersampling:**
- Reduce clase mayoritaria
- Pérdida de información pero más rápido

4. **Stratified Split:**
```python
train_test_split(..., stratify=y)
```
Cada fold mantiene proporción original

5. **Métrica correcta:**
- NO accuracy
- SÍ F1, precision-recall, ROC-AUC

---

### Q9: Feature Importance

**Respuesta modelo:**

**Cómo respondo al cliente:**

1. **Extraigo importancia:**
```python
importances = rf.feature_importances_
top_10 = argsort(importances)[::-1][:10]
```

2. **Visualizo:**
```python
plt.barh(feature_names[top_10], importances[top_10])
```

3. **Interpreto sin tecnicismos:**
- "Las 3 variables más influyentes en la predicción son: X, Y, Z"
- "X contribuye 25% de la decisión total"

**⚠️ Advertencias importantes:**
- Importancia ≠ Causalidad
- Puede haber multicolinealidad
- Features correlacionadas comparten importancia

**Alternativa: SHAP values**
```python
import shap
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
```
Más interpretable y confiable

---

### Q10: Cross-Validation Strategy (Time Series)

**Respuesta modelo:**

**Problema con K-Fold normal:**
- Rompe el orden temporal
- Data leakage: información del futuro filtra al pasado
- No simula verdadero escenario de producción

**Solución: Time Series Split (Forward Chaining)**

```
Split 1: Train[1-100]  → Val[101-110]
Split 2: Train[1-110]  → Val[111-120]
Split 3: Train[1-120]  → Val[121-130]
...
```

Cada fold entrena con datos anteriores, valida en futuro.

**Implementación:**
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
```

**Alternativa: Expanding Window**
Cada fold usa más datos históricos (más realista si hay trend)

---

### Q11: Missing Data Handling (EPAM)

**Respuesta modelo:**

**Paso 1: Entender patrón**
- MCAR (Missing Completely At Random): eliminar ok
- MAR (Missing At Random): imputación sofisticada
- MNAR (Missing Not At Random): cuidado extra

**Paso 2: Decidir según %**
- <5%: eliminar filas
- 5-20%: imputación simple (media/mediana)
- >20%: imputación avanzada (KNN, MICE) o feature indicator

**Paso 3: Implementar**

```python
# Simple: media/mediana
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Avanzada: KNN
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)

# Feature indicator
df['age_missing'] = df['age'].isna().astype(int)
```

**Paso 4: Validar**
- Chequea distribución pre/post imputación
- Compara modelos con/sin missing

---

### Q12: Missing Data in Time Series (EPAM)

**Respuesta modelo:**

**Opciones por tipo de gap:**

1. **Pequeños gaps (<5% datos):**
```python
df['value'].interpolate(method='linear', inplace=True)
```
Interpolación lineal suave

2. **Gaps medianos:**
```python
df['value'].fillna(method='ffill')  # Forward fill
df['value'].fillna(method='bfill')  # Backward fill
```
O combinación

3. **Gaps grandes:**
- Descartar período (si posible)
- Usar seasonal decomposition + ARIMA para estimar
- Feature con season anterior

**⚠️ Nunca:**
- Backward fill en series de tiempo (usa información del futuro)
- Interpolar a través de cambios abruptos (ej: eventos)

**Validación:**
```python
# Verifica estructura temporal preservada
assert df.index.is_monotonic_increasing
```

---

### Q13: K-Means vs DBSCAN (EPAM)

**Respuesta modelo:**

| Aspecto | K-Means | DBSCAN |
|--------|---------|--------|
| **Tipo** | Centroid-based | Density-based |
| **k a priori** | Sí (k clusters) | No (eps, minPts) |
| **Shape clusters** | Esféricos | Arbitrarios |
| **Outliers** | Asigna a cluster | Marca como noise |
| **Velocidad** | Rápido | Más lento |
| **Scaling** | Sensible | Necesario |
| **Parámetros** | k (intuición) | eps, minPts (más difícil) |

**Cuándo usar cada uno:**

**K-Means:**
- Clusters esféricos esperados
- Sabes número de clusters
- Datos limpios
- Velocidad importante

**DBSCAN:**
- Shapes arbitrarios
- Desconoces número de clusters
- Outliers presentes
- Densidades variables

**Ejemplo:**
- Datos de clientes: K-Means (segmentación clara)
- Datos de anomalía: DBSCAN (detecta outliers)

---

### Q14: Clustering Evaluation (EPAM)

**Respuesta modelo:**

**Sin labels (unsupervised metrics):**

1. **Silhouette Score:** [-1, 1]
   - 1: cluster muy denso y separado
   - 0: solapamiento
   - -1: punto en cluster equivocado
   ```python
   from sklearn.metrics import silhouette_score
   score = silhouette_score(X, labels)
   ```

2. **Davies-Bouldin Index:** [0, ∞]
   - Menor es mejor
   - Ratio promedio de compacidad/separación
   ```python
   from sklearn.metrics import davies_bouldin_score
   score = davies_bouldin_score(X, labels)
   ```

3. **Calinski-Harabasz Index:** [0, ∞]
   - Mayor es mejor
   - Ratio separación/compacidad
   ```python
   from sklearn.metrics import calinski_harabasz_score
   score = calinski_harabasz_score(X, labels)
   ```

**Con labels (supervized metrics):**
- Purity, Rand Index, NMI, Adjusted Rand Index

**Visualización:**
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels)
```

---

### Q15: Clustering for Business Problem (EPAM)

**Respuesta modelo:**

**Ejemplo: Segmentación de Clientes de E-commerce**

**Problema:**
- Tenemos 100k clientes
- Objetivo: identificar segmentos para marketing personalizado

**Datos:**
- Frecuencia compra, monto gastado, categorías preferidas, recency
- Sin etiquetas previas

**Preprocesamiento:**
```python
# Escalado (K-Means sensible)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature engineering
features: [frecuencia, monto_promedio, num_categorias, dias_ultima_compra]
```

**Algoritmo:** K-Means
- Razón: clusters esféricos esperados, velocidad importante

**Tuning:**
```python
# Elbow method
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
plt.plot(inertias)
# Elegir k en "codo"
```
Resulta: k=4 óptimo

**Resultados:**
- Cluster 0 (High Value): ~15% clientes, spend promedio $5000/año
- Cluster 1 (Growing): ~25% clientes, spend $1500/año, frecuencia creciente
- Cluster 2 (Dormant): ~40% clientes, último acceso >6 meses
- Cluster 3 (Window Shopper): ~20% clientes, visitan pero no compran

**Impacto business:**
- High Value: retención prioritaria + VIP programs
- Growing: cross-sell, incentivos
- Dormant: win-back campaigns
- Window Shopper: conversion focus

**Learnings:**
- Silhouette score = 0.55 (aceptable)
- Cluster 2 muy grande → puede subdividirse
- Temporal component importante → agregar recency a features

---

## ROUND 3: Deep Learning & Computer Vision (Lunes 10:00-18:00)

### Q16: Neural Networks Basics

**Respuesta modelo:**

**Neurona artificial:**
```
Input: x₁, x₂, x₃, ...
Pesos: w₁, w₂, w₃, ...
Bias: b

Output = Activation(Σ(wᵢ * xᵢ) + b)
```

**Función de Activación:**
- Introduce no-linealidad
- Sin activación (lineal en todas capas): red colapsa a mapeo lineal
- Pierde capacidad de expresar relaciones complejas

**Activaciones comunes:**

| Función | Cuándo | Rango |
|---------|--------|-------|
| ReLU | Capas ocultas (standard) | [0, ∞) |
| Sigmoid | Output binaria | (0, 1) |
| Softmax | Output multi-clase | (0, 1) suma 1 |
| Tanh | Ocasiones | (-1, 1) |

**Ejemplo:**
```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),  # Oculta
    Dense(32, activation='relu'),                       # Oculta
    Dense(10, activation='softmax')                     # Output (10 clases)
])
```

**¿Por qué no lineal en todo?**
Composition de funciones lineales = función lineal
```
f(x) = Linear(Linear(x)) = Linear(x)
```
No ganamos poder expresivo

---

### Q17: Backpropagation Intuición

**Respuesta modelo:**

**Paso 1: Forward Pass**
- Datos fluyen hacia adelante
- Cada neurona computa output
- Output final vs ground truth → Loss

**Paso 2: Calculate Loss**
```
Loss = diferencia(predicción, real)
Ejemplo: MSE = (y_true - y_pred)²
```

**Paso 3: Backward Pass (Backprop)**
- Propagamos el error hacia atrás por la red
- Calculamos gradiente del loss respecto a cada peso

**Paso 4: Update Weights**
```
w_nuevo = w_viejo - learning_rate * gradient
```

**Intuición visual:**
```
Forward:  Input → [Layer1] → [Layer2] → Prediction → Loss
Backward: Loss → [Layer2.grad] → [Layer1.grad] → Update weights
```

**Repetir:** múltiples épocas hasta convergencia

**Analogía:** Descender una montaña en la niebla
- Forward pass: mides altitud (loss)
- Backward pass: calculas pendiente (gradient)
- Update: das un paso downhill

---

### Q18: Convolutional Layers (EPAM)

**Respuesta modelo:**

**¿Qué es una convolución?**

Filtro pequeño (ej: 3×3) desliza sobre imagen
En cada posición: suma elemento-a-elemento (dot product)

```
Imagen:        Filtro:
[1 2 3]        [0.1 0.2]
[4 5 6]  ×     [0.3 0.4]
[7 8 9]
```

**¿Qué detecta?**
- Primeras capas: bordes, texturas (gradientes)
- Capas intermedias: patrones (ojos, nariz)
- Capas finales: objetos completos

**Ventaja sobre fully connected:**

| Aspecto | Fully Connected | Convolución |
|--------|----------------|----|
| Parámetros imagen 224×224 RGB | 150M | 10k |
| Spatial awareness | No | Sí |
| Translation invariance | No | Sí |
| Computable | No (lento) | Sí (rápido) |

**Arquitectura típica CNN:**
```
Input (224×224×3)
  ↓
Conv (32 filtros) → ReLU → MaxPool
  ↓
Conv (64 filtros) → ReLU → MaxPool
  ↓
Flatten
  ↓
Dense (128) → ReLU
  ↓
Dense (10) → Softmax
```

---

### Q19: YOLO vs RCNN (EPAM)

**Respuesta modelo:**

**YOLO (You Only Look Once):**
- Una pasada, una red end-to-end
- Predice región + clase en grid único
- **Velocidad:** Real-time (30-60 FPS)
- **Accuracy:** Bueno (no SOTA)
- **Mejor para:** detección en tiempo real (drones, videos)

**Faster R-CNN (Region-based):**
- Dos fases:
  1. RPN (Region Proposal Network): genera ~2000 regiones candidatas
  2. Clasificador: clasifica cada región
- **Velocidad:** Lento (5-10 FPS)
- **Accuracy:** Excelente (SOTA)
- **Mejor para:** análisis offline, máxima precisión

**Comparación directa:**

| Métrica | YOLO | Faster R-CNN |
|---------|------|-------------|
| Velocidad | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Accuracy | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Pequeños objetos | ⭐⭐ | ⭐⭐⭐⭐ |
| FPS | 30-60 | 5-10 |

**Trade-off:** Speed vs Accuracy

---

### Q20: Transfer Learning (EPAM)

**Respuesta modelo:**

**Pre-trained Model Approach:**

✅ **Ventajas:**
- Features generales ya aprendidas (ImageNet: 1M imágenes)
- Requiere menos datos (500 imágenes suficientes)
- Muy rápido (fine-tuning en horas vs entrenamiento en días)
- SOTA performance típicamente

**Cómo funciona:**
```
ImageNet-trained ResNet50 (feature extractor)
  ↓ (congelar primeras capas)
Custom Dense layers (entrenar estas)
  ↓
Output (10 clases tus datos)
```

**From Scratch Approach:**

✅ **Ventajas:**
- Control total sobre arquitectura
- Si dominio muy diferente (ej: rayos X médicos)

❌ **Desventajas:**
- Requiere MUCHOS datos (millones)
- Entrenar lento (semanas GPU)
- Difícil converger
- Típicamente peor resultado

**Recomendación:**

```
Datos < 1000:       Transfer Learning obligatorio
Datos 1000-10k:     Transfer Learning + fine-tuning
Datos > 100k:       Considera from scratch
Dominio nuevo:      Transfer Learning (best bet)
```

**Caso 500 imágenes de perros:**
```python
# Pre-trained
base_model = ResNet50(weights='imagenet')
base_model.trainable = False  # Congelar

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binaria
])

model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy')
model.fit(X_train, y_train, epochs=20)  # Rápido
```

---

### Q21: Image Segmentation Medical (EPAM)

**Respuesta modelo:**

**Problema:** Segmentar tumores en imágenes CT/MRI

**Arquitectura: U-Net**
```
Input (256×256)
  ↓
Encoder: Conv → ReLU → Conv → MaxPool (downsample)
  ↓
Decoder: Upsample → Conv → ReLU (con skip connections)
  ↓
Output (256×256) = máscara de segmentación
```

**Data Augmentation (domain-aware):**
```python
# ✓ Aplicar
- Rotaciones pequeñas (±15°)
- Flips (vertical/horizontal)
- Slight zoom
- Elastic deformations

# ✗ NO aplicar (rompe estructura médica)
- Cambios de intensidad extremos
- Distorsiones agresivas
```

**Manejo de class imbalance:**
- Mayoría = fondo, minoría = tumor
- Loss: Dice loss, Focal loss (mejor que Cross-entropy)

```python
def dice_loss(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    return 1 - (2 * intersection / union)
```

**Evaluation:**
- Dice coefficient: [0, 1] (1 = perfecto)
- IoU (Intersection over Union)
- Hausdorff distance (shape similarity)
- Validación con radiologists

**Interpretabilidad crítica:**
- Visualiza predicción + ground truth lado a lado
- Heatmaps de activación (CAM)
- Error analysis por radiologist

---

### Q22: Text Preprocessing (EPAM)

**Respuesta modelo:**

**Orden típico:**

1. **Lowercasing:**
```python
text = text.lower()  # "Hello" → "hello"
```

2. **Tokenización:**
```python
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
# "hello world" → ["hello", "world"]
```

3. **Remove special characters & HTML:**
```python
import re
text = re.sub(r'[^a-zA-Z\s]', '', text)
text = re.sub(r'<[^>]+>', '', text)
```

4. **Stopwords removal:**
```python
from nltk.corpus import stopwords
stop = stopwords.words('english')
tokens = [w for w in tokens if w not in stop]
# Quita: "the", "is", "a", "and", ...
```

5. **Stemming vs Lemmatization:**
```python
# Stemming (agresivo): "running" → "run"
from nltk.stem import PorterStemmer
ps = PorterStemmer()
tokens = [ps.stem(w) for w in tokens]

# Lemmatization (lingüístico): "running" → "run"
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()
tokens = [lem.lemmatize(w) for w in tokens]
```

6. **Números y símbolos:**
- Decir: quitar o reemplazar por `<NUM>`, `<UNKNOWN>`

**Nota:** El orden depende del problema
- Sentiment analysis: mantener "!" (emoticones)
- Classification: aggressive cleaning ok

---

### Q23: Sentiment Analysis Model (EPAM)

**Respuesta modelo:**

**Fase 1: Datos**
- Recolectar reviews (user, rating, texto)
- Labeling: 1-2 stars = negativo, 4-5 = positivo (descartar 3)
- Balance dataset (si needed: SMOTE, oversampling)

**Fase 2: Preprocessing**
```python
texts = [clean_text(t) for t in raw_texts]
# Lowercasing, tokenización, stopwords
```

**Fase 3: Features**
Opciones progresivas:

1. **TF-IDF (baseline rápido):**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(max_features=5000)
X = vec.fit_transform(texts)
```

2. **Word Embeddings (Word2Vec, GloVe):**
```python
from gensim.models import Word2Vec
w2v = Word2Vec(sentences=tokens)
# Cada palabra → vector denso 100-d
```

3. **BERT (pre-trained Transformer):**
```python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
embeddings = model(token_ids)
# Sentence embedding = mean pooling
```

**Fase 4: Modelo**
Progressión:

1. **Logistic Regression (TF-IDF)** → 0.80 F1
2. **LSTM (embeddings)** → 0.86 F1
3. **BERT fine-tuning** → 0.92 F1

**Fase 5: Evaluation**
```python
# Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
```

**Deployment:**
```python
# Inference rápido
def predict_sentiment(text):
    tokens = clean_text(text)
    embedding = bert_encode(tokens)
    prob = model.predict(embedding)
    return "positive" if prob > 0.5 else "negative"
```

---

### Q24: Word2Vec & Embeddings (Original)

**Respuesta modelo:**

**Embedding = representación densa en espacio vectorial**

```
Palabra → Vector (ej: 100 dimensiones)
"king" → [0.2, -0.5, 0.1, ...]
"queen" → [0.1, -0.6, 0.15, ...]
"man" → [0.3, -0.4, 0.05, ...]
"woman" → [0.0, -0.7, 0.2, ...]
```

**¿Cómo Word2Vec aprende?**

Objetivo: palabras que aparecen juntas en contexto tienen embeddings similares

```
"King sat on the throne"
     ↑        ↑

word2vec aprende:
- "king" y "throne" contexto similar
- Sus vectores cercanos en espacio
```

**La magia algebraica:**

```
king - man + woman ≈ queen

Porque:
- "king" = concepto_realeza + concepto_masculino
- "man" = concepto_masculino
- "woman" = concepto_masculino convertido a femenino
- Restar "masculino" y sumar "femenino" →realeza femenina = "queen"
```

**Implementación:**
```python
from gensim.models import Word2Vec

model = Word2Vec(sentences=corpus, vector_size=100, window=5)

# Palabra más similar a "king"
similar = model.wv.most_similar("king", topn=5)

# Operación algebraica
result = model.wv.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=1
)
# Resultado: "queen"
```

---

### Q25: Transformers vs RNNs (EPAM)

**Respuesta modelo:**

**RNN Tradicional (LSTM, GRU):**

❌ **Limitaciones:**
- Procesa secuencia paso a paso (secuencial)
- Información antigua se olvida (vanishing gradient)
- No puede paralelizarse (lento)
- Máxima distancia eficaz: ~500 tokens

**Ejemplo:**
```
Texto: "The cat sat on the mat"
RNN: procesa palabra a palabra
[The] → [cat] → [sat] → [on] → [the] → [mat]
(12 pasos)
```

**Transformer:**

✅ **Ventajas:**
- Procesa TODO simultáneamente (parallelizable)
- Atención: cada palabra "ve" todas las otras
- Escala a textos largos (miles de tokens)
- Más rápido entrenar
- SOTA en casi todas tasks

**Arquitectura:**
```
Self-Attention: "¿Cuál es la palabra más relevante para cada palabra?"

"The dog chased the cat"
 ↑    ↑      ↑      ↑   ↑
"the" ve: "dog" (sujeto), "chased" (verbo), "cat" (objeto)
Calcula relevancia dinámicamente
```

**Comparación directa:**

| Aspecto | RNN | Transformer |
|---------|-----|-------------|
| Velocidad entrenamiento | Lento | Rápido (parallelizable) |
| Long-range dependencies | Débil | Fuerte |
| Memoria | Comprimida en estado | Explícita (atención) |
| SOTA | No | Sí |
| Ejemplos | LSTM, GRU | BERT, GPT, T5 |

**Cuándo usar:**
- Transformers: casi siempre (2023+)
- RNNs: datos muy pequeños, latency crítica

---

### Q26: Attention Mechanism (EPAM)

**Respuesta modelo:**

**Problema que resuelve:**

RNN pierde contexto en textos largos:
```
"The bank executive walked into the bank..."
                                    ↑
RNN "olvida" que "bank" aquí significa institución financiera
(pasó hace 10 palabras)
```

**Atención: solución**

Para cada palabra, calcula "qué palabras son relevantes":

```
Mecanismo Scaled Dot-Product Attention:

Query (Q): "¿Qué busco?"      [Embedding actual]
Key (K):   "Qué puedo ofrecer" [Embeddings todas palabras]
Value (V): "Mi contenido"      [Embeddings todas palabras]

Relevance = softmax(Q · K^T / √d)
Output = Relevance · V
```

**Ejemplo visual:**
```
"The bank executive walked..."
                    ↑
Al procesar "walked", atención calcula:
- "The" (artículo): relevancia baja
- "bank" (sujeto): relevancia alta
- "executive" (quién camina): relevancia muy alta
- "walked" (yo mismo): relevancia media

Output = weighted average de todos
```

**Multi-Head Attention:**
```
En lugar de 1 atención:
- Head 1: ¿Cuál es el sujeto?
- Head 2: ¿Cuál es el verbo?
- Head 3: ¿Quién hace qué?
- ...8 cabezas

Combina todas las perspectivas
```

**Ventaja:**
- Cada palabra "ve" todas las otras
- Explícito qué importa (interpretable)
- Largo alcance sin vanishing gradient

---

### Q27: ARIMA Forecasting (EPAM)

**Respuesta modelo:**

**ARIMA(p, d, q):**

- **p (AR):** Autoregressive — datos anteriores
- **d (I):** Integrated — diferencing para stationarity
- **q (MA):** Moving Average — errores anteriores

**Pasos para forecasting:**

**1. Visualiza serie:**
```python
import matplotlib.pyplot as plt
plt.plot(df['sales'])
plt.show()
```
¿Hay trend? ¿Seasonality? ¿Outliers?

**2. Test stationarity (Augmented Dickey-Fuller):**
```python
from statsmodels.tsa.stattools import adfuller
result = adfuller(data)
if result[1] > 0.05:  # p-value
    # No estacionaria, necesita diferencing
```

**3. Diferenciar si no estacionaria:**
```python
data_diff = data.diff()  # Primera diferencia
# Repetir test
```

**4. Elegir p, q (ACF/PACF plots):**
```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2)
plot_acf(data_diff, ax=axes[0], lags=40)
plot_pacf(data_diff, ax=axes[1], lags=40)
plt.show()
# ACF tail-off → p = ? (donde PACF cuts)
# PACF tail-off → q = ? (donde ACF cuts)
```

**5. Fit ARIMA:**
```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(data, order=(2, 1, 2))
results = model.fit()
print(results.summary())
```

**6. Diagnósticos:**
```python
results.plot_diagnostics(figsize=(12, 8))
plt.show()
# Residuals ≈ white noise? Good
```

**7. Forecast:**
```python
forecast = results.get_forecast(steps=24)  # Próximos 24 períodos
confidence = forecast.conf_int()

plt.plot(data)
plt.plot(forecast.predicted_mean, label='Forecast')
plt.fill_between(confidence.index, 
                 confidence.iloc[:, 0],
                 confidence.iloc[:, 1], alpha=0.3)
plt.legend()
plt.show()
```

---

### Q28: Time Series Features (EPAM)

**Respuesta modelo:**

**Feature engineering para time series:**

1. **Lags (autoregresión):**
```python
df['sales_lag1'] = df['sales'].shift(1)
df['sales_lag7'] = df['sales'].shift(7)  # Una semana atrás
```

2. **Rolling statistics:**
```python
df['sales_rolling_mean_7'] = df['sales'].rolling(7).mean()
df['sales_rolling_std_7'] = df['sales'].rolling(7).std()
```

3. **Seasonal features:**
```python
df['month'] = df.index.month
df['day_of_week'] = df.index.dayofweek
df['quarter'] = df.index.quarter

# One-hot encoding
df = pd.get_dummies(df, columns=['month'])
```

4. **Trend:**
```python
from sklearn.linear_model import LinearRegression
X_time = np.arange(len(df)).reshape(-1, 1)
lr = LinearRegression()
lr.fit(X_time, df['sales'])
df['trend'] = lr.predict(X_time)
```

5. **External features:**
```python
# Temperatura, competencia, promociones, vacaciones
df['temperature'] = external_weather_data
df['promo'] = [1 if date in promo_dates else 0 for date in df.index]
```

6. **Differencing (elimina trend/seasonality):**
```python
df['sales_diff'] = df['sales'].diff()
df['sales_seasonal_diff'] = df['sales'].diff(12)  # 12 meses atrás
```

**Validación:**
```python
# Chequea correlación con target
df.corr()['sales'].sort_values(ascending=False)
```

---

### Q29: Seasonality in Time Series (EPAM)

**Respuesta modelo:**

**Identificar seasonality:**

1. **Gráfico:**
```python
plt.figure(figsize=(14, 4))
plt.plot(df['sales'])
plt.show()
# ¿Patrón repetido cada período?
```

2. **ACF plot:**
```python
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df['sales'], lags=40)
# Picos regulares → Seasonality presente
# Si pico en lag 12 → seasonality = 12 meses
```

3. **Seasonal decomposition:**
```python
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df['sales'], model='additive', period=12)

fig, axes = plt.subplots(4, 1, figsize=(14, 10))
result.observed.plot(ax=axes[0], title='Observed')
result.trend.plot(ax=axes[1], title='Trend')
result.seasonal.plot(ax=axes[2], title='Seasonal')
result.resid.plot(ax=axes[3], title='Residual')
```

**Direcciones de tratamiento:**

1. **Seasonal differencing:**
```python
df['sales_seasonal_diff'] = df['sales'].diff(12)
# Luego fit ARIMA en datos diferenciados
```

2. **SARIMA (Seasonal ARIMA):**
```python
model = SARIMAX(df['sales'], 
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12))  # (P, D, Q, s)
results = model.fit()
```

3. **Prophet (meta-algorithm):**
```python
from fbprophet import Prophet
m = Prophet(yearly_seasonality=True)
m.fit(df)  # Detecta seasonality automáticamente
forecast = m.make_future_dataframe(periods=12)
forecast = m.predict(forecast)
```

4. **Manual feature:**
```python
df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
# Captura circularidad del mes (diciembre ≈ enero)
```

**Validación:**
```python
# Después de remover seasonality:
result.resid.plot()  # ¿White noise?
```

---

### Q30: Recommendation System for Streaming (EPAM)

**Respuesta modelo:**

**Sistema para Netflix-like:**

**1. Cold-start problem:**
- Usuarios nuevos: sin histórico
- Items nuevos: sin ratings

**Soluciones:**
- Recomendaciones populares iniciales
- Preguntar preferencias (onboarding)
- Mezclar: contenido popular + exploratorio

**2. Tipos de algoritmos:**

A) **Collaborative Filtering:**
```python
# User-based: "usuarios similares ven contenido similar"
similar_users = find_similar_users(user_id, similarity='cosine')
recommendations = content_watched_by(similar_users)

# Item-based: "usuarios que ven A también ven B"
similar_items = find_similar_items(item_id, ratings_matrix)
recommendations = similar_items

# Matrix factorization (SVD):
U, sigma, VT = SVD(user_item_matrix)
predictions = U[:, :k] @ sigma[:k, :k] @ VT[:k, :]
```

B) **Content-based:**
```python
# "Te recomendamos películas similares a las que ves"
item_features = extract_features(movie)  # género, actor, director
similar_items = find_similar(item_features, all_movies)
```

C) **Hybrid:**
```python
# Combina ambas
cf_score = collab_filtering_score(user, item)
cb_score = content_based_score(user, item)
final_score = 0.7 * cf_score + 0.3 * cb_score
```

**3. Real-time constraints:**
- Latency: <200ms típicamente
- Soluciones:
  - Precomputar nearestneighbors
  - Cache popular items
  - Serving layer rápido (Redis)

**4. Evaluation:**
- **Offline:** Precision@10, Recall@10, NDCG, MAP
- **Online:** Click-through rate (CTR), watch-time, retention

```python
def precison_at_k(actual, predicted, k):
    predicted_k = predicted[:k]
    hits = len(set(actual) & set(predicted_k))
    return hits / k

def recall_at_k(actual, predicted, k):
    predicted_k = predicted[:k]
    hits = len(set(actual) & set(predicted_k))
    return hits / len(actual)

def ndcg_at_k(relevance_scores, k):
    # Normalized Discounted Cumulative Gain
    dcg = sum(rel / np.log2(i+2) for i, rel in enumerate(relevance_scores[:k]))
    idcg = sum(1 / np.log2(i+2) for i in range(min(len(relevance_scores), k)))
    return dcg / idcg
```

**5. Business KPIs:**
- Watch-time / user
- Retention (% users vuelven)
- Diversity (no solo blockbusters)
- Serendipity (sorpresa positiva)

---

## ROUND 4: Advanced Scenarios (Martes 11:00-15:00)

### Q31: End-to-End ML Pipeline

**Respuesta modelo:**

**Proyecto ejemplo: Churn Prediction (telecomunicaciones)**

**1. Problem Statement:**
- Objetivo: Predecir qué clientes se irán
- Métrica primaria: Recall ≥ 85% (no queremos perder clientes)
- Negocio: Intervención cuesta $100, CLV = $500

**2. EDA:**
```python
df.info()            # Tipos, nulos
df.describe()        # Estadísticas
df.isnull().sum()    # % missing
df['churn'].value_counts()  # Distribución
```
Hallazgo: 73% clase negativa, 27% positiva

**3. Preprocessing:**
```python
# Missing: Internet_service 3% → fill mode
df['Internet_service'].fillna(df['Internet_service'].mode()[0], inplace=True)

# Outliers: MonthlyCharges tiene uno (9999) → remove
df = df[df['MonthlyCharges'] < 1000]

# Categorical: Contract → one-hot
df = pd.get_dummies(df, columns=['Contract'], drop_first=True)

# Scaling: Tenure, MonthlyCharges → StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['Tenure', 'MonthlyCharges']] = scaler.fit_transform(df[['Tenure', 'MonthlyCharges']])
```

**4. Feature Engineering:**
```python
# Ratio: MonthlyCharges / Tenure
df['charge_per_month_tenure'] = df['MonthlyCharges'] / (df['Tenure'] + 1)

# Interaction: Tech_support * Internet_service
df['tech_support_internet'] = df['Tech_support'] * df['Internet_service']
```

**5. Modeling:**
```python
# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Baseline: Dummy
baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train, y_train)
print(f"Baseline Recall: {recall_score(y_test, baseline.predict(X_test))}")  # 0% (siempre negativo)

# Model 1: Logistic Regression
lr = LogisticRegression(class_weight='balanced')
lr.fit(X_train, y_train)
lr_recall = recall_score(y_test, lr.predict(X_test))
print(f"LR Recall: {lr_recall}")  # 72%

# Model 2: Random Forest (tuned)
from sklearn.model_selection import GridSearchCV
rf = RandomForestClassifier(class_weight='balanced')
params = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
grid = GridSearchCV(rf, params, cv=5, scoring='recall')
grid.fit(X_train, y_train)
rf_recall = recall_score(y_test, grid.best_estimator_.predict(X_test))
print(f"RF Recall: {rf_recall}")  # 82%

# Model 3: XGBoost
xgb_model = xgb.XGBClassifier(scale_pos_weight=(1-0.27)/0.27)
xgb_model.fit(X_train, y_train)
xgb_recall = recall_score(y_test, xgb_model.predict(X_test))
print(f"XGB Recall: {xgb_recall}")  # 86% ✓
```

**6. Evaluation:**
```python
# XGB gana (recall 86%)
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba)}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(10))
```

**7. Deployment:**
```python
import joblib
joblib.dump(xgb_model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Inference
def predict_churn(customer_data):
    scaled = scaler.transform(customer_data)
    prob = xgb_model.predict_proba(scaled)[0, 1]
    return "High risk" if prob > 0.5 else "Low risk"
```

**Error cometido:** Escalé features ANTES de train/test split
**Cómo lo resolví:** Moví scaler.fit() DESPUÉS del split
**Learnings:** Data leakage es fácil, piensa en data flow

---

### Q32: Data Leakage

**Respuesta modelo:**

**Data leakage = información del futuro filtra al pasado/train**

**Ejemplos clásicos:**

1. **Scaling antes de split:**
```python
❌ X_scaled = scaler.fit_transform(X)
   X_train, X_test = train_test_split(X_scaled)
   # Test information en scaler (min/max de test)

✅ X_train, X_test = train_test_split(X)
   scaler.fit(X_train)  # Solo train
   X_train_scaled = scaler.transform(X_train)
   X_test_scaled = scaler.transform(X_test)
```

2. **Target encoding sin CV:**
```python
❌ target_mean = df.groupby('category')['target'].mean()
   df['category_encoded'] = df['category'].map(target_mean)
   # Group mean incluye test data

✅ for fold_idx, (train_idx, test_idx) in cv.split(X):
       # Calcular mean solo en train fold
       train_mean = df.iloc[train_idx].groupby('category')['target'].mean()
       # Aplicar a test
       test_encoded = df.iloc[test_idx]['category'].map(train_mean)
```

3. **Time series: usando futuro:**
```python
❌ df['future_feature'] = df['target'].shift(-7)  # Futuro
   model.fit(X[df['future_feature'].notna()], y)

✅ df['past_feature'] = df['target'].shift(7)  # Pasado
```

4. **Feature engineering en toda data:**
```python
❌ df['outlier_removed'] = df[df['feature'] < df['feature'].quantile(0.99)]
   # Quantile = info de TODA la data

✅ q99 = X_train['feature'].quantile(0.99)
   X_train = X_train[X_train['feature'] < q99]
   X_test = X_test[X_test['feature'] < q99]  # Usar threshold train
```

**Cómo detectar:**
- Test performance >> validation performance
- Training rápido, test lento a predecir
- Métricas perfectas (99% accuracy sospechoso)

**Prevención:**
- Pipeline sklearn (automatiza)
- Separa train/test PRIMERO
- Fit transformaciones SOLO en train

---

### Q33: Model Monitoring

**Respuesta modelo:**

**Modelo en producción necesita vigilancia:**

**1. Prediction monitoring:**
```python
# ¿Cambió distribución de predicciones?
pred_distribution_today = model.predict_proba(X_today)[:, 1]
pred_distribution_baseline = pred_distribution_baseline  # de entrenamiento

# Kolmogorov-Smirnov test
from scipy.stats import ks_2samp
statistic, p_value = ks_2samp(pred_distribution_today, pred_distribution_baseline)
if p_value < 0.05:
    alert("Prediction distribution shifted")
```

**2. Feature monitoring (Input drift):**
```python
# ¿Cambió distribución de features?
for feature in features:
    today = X_today[feature]
    baseline = X_train[feature]
    statistic, p_value = ks_2samp(today, baseline)
    if p_value < 0.05:
        alert(f"{feature} has drifted")
```

**3. Performance monitoring (si tienes labels delayed):**
```python
# Cada semana: evalúa en nuevos datos
accuracy_today = accuracy_score(y_true_today, y_pred_today)
if accuracy_today < baseline_accuracy - 0.05:
    alert("Accuracy degraded")
```

**4. Concept drift:**
```python
# La relación entre X e y cambió
# Ejemplo: en pandemia, patrón de compra cambió
# Detector: ADWIN (Adaptive Windowing)

from river import drift
detector = drift.ADWIN()
for i, (x, y) in enumerate(stream):
    pred = model.predict(x)
    error = abs(pred - y)
    detector.update(error)
    if detector.drift_detected:
        alert("Concept drift detected")
```

**5. Alertas útiles:**
```python
# Monitoreo dashboard
metrics = {
    'prediction_volume': n_predictions_today,
    'avg_prediction_score': np.mean(y_pred_proba),
    'std_prediction_score': np.std(y_pred_proba),
    'feature_mean': X_today.mean(),
    'latency_p95': latency_95_percentile,
    'cache_hit_rate': hits / total
}

# Alertas
if prediction_volume < expected * 0.8:
    alert("Low prediction volume")
if latency_p95 > 200ms:
    alert("High latency")
```

**6. Retraining strategy:**
```python
# Opción 1: Scheduled (cada semana)
if today.day_of_week == 0:  # Lunes
    retrain_model()

# Opción 2: Triggered (si degrada)
if accuracy_today < baseline - 0.05:
    retrain_model()

# Opción 3: Continuous (online learning)
for new_sample in stream:
    model.partial_fit(new_sample)  # Actualiza incremental
```

---

### Q34: Interpreting Black Box

**Respuesta modelo:**

**Cliente pregunta: "¿Por qué rechazaste mi crédito?"**

No puedo mostrar árbol XGBoost (incomprensible).

**Solución 1: SHAP values**
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[idx])

# Visualizar
shap.plots.waterfall(shap_values[idx])
```
Muestra: "Tu ratio deuda/ingresos sumó -0.3 a la decisión"

**Solución 2: LIME (Local Interpretable Model-agnostic Explanations)**
```python
from lime.tabular import LimeTabularExplainer
explainer = LimeTabularExplainer(X_train, feature_names=feature_names)
explanation = explainer.explain_instance(X_test[idx], model.predict_proba)
explanation.show_in_notebook()
```
Aproxima el modelo localmente con regresión lineal interpretable

**Solución 3: Feature importance en predicción específica**
```python
# Qué features movieron la aguja
permutation_importance = {}
baseline_pred = model.predict_proba(X_test[idx])[0, 1]

for feature in features:
    X_permuted = X_test[idx].copy()
    X_permuted[feature] = np.random.choice(X_train[feature])
    new_pred = model.predict_proba(X_permuted)[0, 1]
    importance = abs(baseline_pred - new_pred)
    permutation_importance[feature] = importance

# Top 3 que movieron la predicción
top_features = sorted(permutation_importance, key=lambda x: permutation_importance[x], reverse=True)[:3]
```

**Explicación al cliente (no técnica):**
```
"Tu aplicación fue denegada porque:

1. Tu deuda actual es 45% de tus ingresos (límite: 30%)
2. Tu historial crediticio tiene 2 pagos atrasados en los últimos 2 años
3. Tu ingresos varían mucho (desviación: 25%)

Recomendación: Reduce deuda o incrementa ingresos estables. Espera 6 meses sin atrasos."
```

---

### Q35: Balancing Speed vs Accuracy

**Respuesta modelo:**

**Contexto:** Cliente necesita predicciones <100ms
**XGBoost actual:** 95% accuracy, 500ms latency

**Decisión depende del caso:**

**Caso A: Ad placement (ms críticos)**
- Sacrifico accuracy por speed
- Estrategias:
  1. Logistic Regression en lugar de XGBoost
  2. Feature selection: top 20 features solo
  3. Quantization: pesos de 32-bit a 16-bit
  4. Serving: modelo en memoria (Redis), no disco

```python
# Resultado: 85% accuracy, 10ms latency
```

**Caso B: Fraud detection (segundos ok)**
- Balance
- Estrategias:
  1. XGBoost con hiperparámetros livianos (max_depth=5)
  2. Batch prediction (procesar lotes)
  3. Caché de predicciones comunes

```python
# Resultado: 92% accuracy, 150ms latency
```

**Caso C: Análisis offline (no crítico)**
- Maximizo accuracy
- Usar ensemble de XGBoost + LGBM + CatBoost

```python
# Resultado: 97% accuracy, 5s latency (ok para batch)
```

**Técnicas generales para speed:**

1. **Feature selection:**
```python
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=20)
X_selected = selector.fit_transform(X_train, y_train)
```

2. **Model distillation (Student-Teacher):**
```python
# Teacher: XGBoost (lento, preciso)
y_soft = teacher.predict_proba(X_train)[:, 1]

# Student: Logistic Regression (rápido)
student = LogisticRegression()
student.fit(X_train, y_soft)  # Aprende de probabilidades blandas
```

3. **Quantization:**
```python
# Convertir Float32 → Float16 / Int8
model_quantized = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear})
```

4. **Caching + índices:**
```python
# Cache predicciones comunes
cache = {}
def predict_cached(customer_id):
    if customer_id in cache:
        return cache[customer_id]
    else:
        pred = model.predict([customer_id])
        cache[customer_id] = pred
        return pred
```

---

### Q36: Collaborative Filtering Pros/Cons (EPAM)

**Respuesta modelo:**

**Collaborative Filtering = "usuarios similares ven contenido similar"**

**✅ Ventajas:**

1. **No necesita metadata:**
   - Funciona sin descripciones de items
   - Netflix = solo ratings, sin sinopsis

2. **Descubre relaciones ocultas:**
   - Usuario A vio: acción, drama
   - Usuario B vio: acción, drama, + thriller
   - Recomendamos thriller a A (patrón emergente)

3. **Serendipity:**
   - Recomienda items que usuario no buscaría explícitamente
   - Sorpresas positivas = engagement

❌ **Desventajas:**

1. **Cold-start:**
   - Usuario nuevo: sin histórico, no hay "similares"
   - Item nuevo: sin ratings, no aparece en recomendaciones
   - Solución: hybrid + contenido popular

2. **Sparsity:**
   - Usuario típico solo ve ~0.001% de items
   - Matriz usuario-item: 99.9% vacía
   - Hace patterns débiles

3. **Popular bias:**
   - Tiende a recomendar solo blockbusters
   - Nicho content no emerge

4. **Escalabilidad:**
   - Millones de usuarios × millones de items
   - Cálculo de similitud O(n²)
   - Solución: matrix factorization

**Variantes:**

- **User-based:** similitud entre usuarios
- **Item-based:** similitud entre items
- **Matrix Factorization (SVD):** factoriza rating matrix
- **Deep Learning:** autoencoders, neural CF

---

### Q37: Evaluating Recommendation Systems (EPAM)

**Respuesta modelo:**

**Métricas offline (sin usuarios reales):**

```python
# Precision@K: % de top-K recomendaciones que el usuario realmente le gusta
precision_at_10 = (items_liked ∩ top_10_recommended) / 10

# Recall@K: % de items que le gusta que aparecen en top-K
recall_at_10 = (items_liked ∩ top_10_recommended) / len(items_liked)

# NDCG (Normalized Discounted Cumulative Gain): considera ranking
# Penaliza items buenos que aparecen tarde

# MAP (Mean Average Precision): promedio de precisiones en cada K
```

**Implementación:**
```python
from sklearn.metrics import ndcg_score

y_true = [0, 0, 1, 1, 0]  # Items liked (en orden ranking)
ndcg = ndcg_score([y_true], [y_true])

# Coverage: % items recomendados (diversidad)
coverage = len(unique_items_recommended) / len(all_items)
```

**Métricas online (con usuarios reales):**

```python
# CTR (Click-Through Rate): % de recomendaciones clickeadas
ctr = clicks / impressions

# Conversion: % que llevó a compra
conversion = purchases / impressions

# Watch-time: minutos vistos (Netflix)
watch_time_per_user = total_minutes / num_users

# Retention: % usuarios que vuelven semana siguiente
retention = users_week_2 / users_week_1
```

**A/B Testing:**
```python
# Control: recomendador actual
# Test: recomendador nuevo

# Métrica: CTR
control_ctr = 0.05
test_ctr = 0.055

# Significancia estadística
from scipy.stats import chi2_contingency
contingency_table = [[control_clicks, control_impressions - control_clicks],
                      [test_clicks, test_impressions - test_clicks]]
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

if p_value < 0.05:
    print("Test es significativamente mejor")
```

**Coverage vs Accuracy:**
```
Todos recomiendan Top-1000 blockbusters:
- Precision alta
- Coverage baja (rest 9M items ignored)

Algoritmo que descubre nicho:
- Precision media
- Coverage alta
- Mejor para negocio (long-tail)
```

---

### Q38: Real-Time Recommendations Challenges (EPAM)

**Respuesta modelo:**

**Desafíos en rec-systems real-time:**

1. **Latency (<200ms típico):**
   - Calcular similitud de usuario a 100M items = imposible
   - Soluciones:
     - Precompute nearest neighbors (top-1000)
     - Índices rápidos (FAISS, Annoy)
     - Cache en Redis

2. **Scalability (M usuarios × M items):**
   - Matriz densa: imposible en RAM
   - Soluciones:
     - Matrix factorization (baja dimensión)
     - Sharding (particionar usuarios/items)
     - Distributed computing (Spark)

3. **Freshness (data cambio rápido):**
   - Items nuevos no tienen ratings
   - Gustos cambian (moda)
   - Soluciones:
     - Recompute clusters cada hora
     - Weights exponencial (nuevo más importante)
     - Hybrid: popular + collaborative

4. **Cold-start:**
   - Usuario nuevo: sin histórico
   - Item nuevo: sin ratings
   - Soluciones:
     - Onboarding: preguntar preferencias
     - Popular items inicialmente
     - Content-based + metadata

5. **A/B testing con dinámica rápida:**
   - Resultados varían por día (efecto novedad)
   - Sample size grande
   - Soluciones:
     - Test mínimo 2 semanas
     - Metricas suavizadas (7-day moving avg)
     - Bandit algorithms (explore-exploit)

**Arquitectura real-time típica:**

```
Request → Serving Layer (Redis cache)
          ↓
          Hit? → Return cached
          ↓ Miss
          Ranking Service (compute top-K)
          ↓
          Collaboration matrix (precomputed)
          ↓
          Return recommendations
```

**Bandit algorithms (explore-exploit):**
```python
# ε-greedy: 80% best, 20% explore
if random() < 0.2:
    recommend_random()
else:
    recommend_best()

# Thompson Sampling: probabilístico, bayesiano
```

---

### Q39: Overlapping Clusters (EPAM)

**Respuesta modelo:**

**Problema:** Puntos entre dos clusters
```
Cluster A (rojo)  Cluster B (azul)
      ◯  ◯              ◯  ◯
       ◯  ◯◯◯◯◯◯◯◯◯◯  ◯
        ◯◯◯◯◯  ?  ◯◯◯◯◯
```

**Soluciones:**

1. **Soft clustering (Fuzzy C-Means):**
```python
from skfuzzy import cmeans
cntr, u, u0, d, jm, p, fpc = cmeans(
    X.T, c=3, c_iterations=100, error=0.005
)
# u = matriz de membresía [0, 1]
# Cada punto: probabilidad de pertenecer a cada cluster
```

2. **Gaussian Mixture Models (probabilístico):**
```python
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
responsibilities = gmm.predict_proba(X)  # Soft assignments
labels = gmm.predict(X)  # Hard assignments
```

3. **Dimensionality reduction primero:**
```python
# Reducir a 2D para visualizar/separar mejor
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# Luego cluster
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X_2d)
```

4. **DBSCAN (detecta overlaps automáticamente):**
```python
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)
# label = -1: outlier (punto en overlap)
```

**Validación de overlaps:**
```python
# Silhouette: negativo en overlaps
from sklearn.metrics import silhouette_samples
silhouette_vals = silhouette_samples(X, labels)
overlap_points = silhouette_vals < 0
print(f"Overlapping points: {np.sum(overlap_points)}")
```

---

### Q40: Model Degradation & Retraining (FINAL)

**Respuesta modelo:**

**Señales de degradación:**

1. **Concept drift (relación X-y cambió):**
   ```python
   # Accuracy baja con el tiempo
   accuracy_week1 = 0.92
   accuracy_week4 = 0.87
   # ¿Cambió el patrón?
   ```

2. **Data drift (distribución X cambió):**
   ```python
   # Ejemplos nuevos ↔ training distribution
   # Ej: pandemia cambió patrón de compra
   
   from scipy.stats import ks_2samp
   stat, p = ks_2samp(X_train['age'], X_today['age'])
   if p < 0.05:
       print("Data drift detected")
   ```

3. **Performance degradation:**
   ```python
   # Métrica cae abajo threshold
   if accuracy_today < 0.85:  # Threshold
       flag = True
   ```

**¿Cuándo reentrenar?**

**Opción 1: Scheduled (regular)**
```python
# Reentrenar cada semana
if today == 'Monday':
    retrain_model()
```

**Opción 2: Triggered (on demand)**
```python
# Reentrenar si degrada
if accuracy_today < baseline - 0.05:
    retrain_model()
```

**Opción 3: Continuous (online learning)**
```python
# Actualizar modelo incremental
for new_sample in daily_stream:
    model.partial_fit([new_sample])  # SGDClassifier, etc.
```

**¿Cuál elegir?**
- Alta volatilidad: Triggered o Continuous
- Bajo cambio: Scheduled (weekly/monthly)
- Crítico: Continuous (banking, medical)

**Validar modelo nuevo:**
```python
# Entrenar en datos históricos
# Evaluar en test set held-out

new_model.fit(X_train_new, y_train_new)
new_accuracy = new_model.score(X_test_held_out, y_test_held_out)

# Solo cambiar si mejora
if new_accuracy > current_accuracy:
    deploy(new_model)
    current_accuracy = new_accuracy
else:
    print("New model not better, keeping current")
```

**Rollback strategy:**
```python
# Si algo falla en producción:
# 1. Detectar rápido (alertas)
# 2. Revert al modelo anterior
# 3. Investigar qué pasó
# 4. Retrain con más cuidado

models_history = [model_v1, model_v2, model_v3_new]
current_version = 2

if accuracy_v3_new < 0.80:
    current_version = 2  # Rollback
    alert("Rolled back to v2")
```

---

## ✅ FIN — 40 Respuestas

Usa este documento como referencia. Adapta tus respuestas a tu estilo, pero asegúrate de cubrir los conceptos clave de cada pregunta.

**Éxito en EPAM.** 🎯

