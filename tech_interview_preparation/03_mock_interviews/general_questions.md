# 🎤 40 Mock Interview Questions — EPAM Data Scientist

**Fuentes:**
- 20 preguntas originales (conceptos generales)
- 20 preguntas del documento oficial de preparación EPAM

**Total: 40 preguntas**  
**Formato:** Responde en chat, luego documento final compilado

---

## 🎬 ROUND 1: Warm-up Fundamentals (Jueves 21:00) — Q1-Q5

### Q1: Overfitting Basics
**Dificultad:** ⭐  
**Tema:** ML Fundamentals

*"Explica qué es overfitting. ¿Cómo lo detectas? ¿Y cómo lo prevendes? Dame un ejemplo concreto."*

**Expected:**
- Definición clara: modelo memoriza en lugar de generalizar
- Detectar: training loss bajo, validation loss alto
- Prevenir: regularización, más datos, simplificar modelo
- Ejemplo: modelo con 1000 features para 100 muestras

---

### Q2: Train/Val/Test Split
**Dificultad:** ⭐⭐  
**Tema:** ML Fundamentals

*"¿Por qué necesitamos separar en train, validation y test? ¿Qué pasa si usas el mismo conjunto para entrenar y evaluar?"*

**Expected:**
- Train: aprende los pesos
- Validation: tuning de hiperparámetros (no contaminar test)
- Test: evaluación final honesta
- Si usas lo mismo: inflación de métricas, overfitting invisible

---

### Q3: Bias-Variance Tradeoff
**Dificultad:** ⭐⭐  
**Tema:** ML Fundamentals

*"¿Qué es el tradeoff bias-variance? Dibuja mentalmente la curva. ¿Cómo se manifiesta en un modelo real?"*

**Expected:**
- Bias alto: modelo muy simple, underfitting
- Variance alta: modelo muy complejo, overfitting
- Sweet spot en el medio
- Ejemplo: polinomio grado 1 vs grado 10

---

### Q4: Precision vs Recall (EPAM Official)
**Dificultad:** ⭐⭐  
**Tema:** Evaluation Metrics

*"Explain the difference between precision and recall."*

**Expected:**
- Precision = TP / (TP + FP) — "De lo que predije positivo, cuánto acerté"
- Recall = TP / (TP + FN) — "De lo positivo real, cuánto detecté"
- Trade-off: aumentar uno baja el otro
- Ejemplos: fraude (recall), spam (precision)

---

### Q5: Overfitting Prevention (EPAM Official)
**Dificultad:** ⭐⭐  
**Tema:** ML Fundamentals

*"What is overfitting, and how can you prevent it?"*

**Expected:**
- Definición: memorizar datos
- Síntomas: train loss bajo vs validation loss alto
- Soluciones:
  - Regularización (L1, L2)
  - Early stopping
  - Más datos / data augmentation
  - Reducir complejidad
  - Cross-validation

---

## 🎯 ROUND 2: Model Selection & Data (Viernes 11:00-15:00) — Q6-Q15

### Q6: Precision vs Recall (Original)
**Dificultad:** ⭐⭐  
**Tema:** Evaluation Metrics

*"Tienes un modelo que detecta fraude bancario. ¿Optimizas para precision o recall? ¿Por qué?"*

**Expected:**
- Contexto importa: falsos positivos vs falsos negativos
- Fraude: recall > precision (no queremos fraude sin detectar)
- Cost of false positive vs false negative
- Explicar trade-off: si subes recall, baja precision

---

### Q7: Choosing Between Models
**Dificultad:** ⭐⭐⭐  
**Tema:** Model Development

*"Tienes un dataset de 100k registros. Necesitas elegir entre Logistic Regression, Random Forest y XGBoost. ¿Cómo decides? ¿Qué criterios usas?"*

**Expected:**
- Interpretabilidad: LR > RF ≈ XGB
- Performance: XGB > RF > LR (típicamente)
- Data size: LR con 100k está bien, XGB es overkill
- Feature engineering needed: XGB requiere menos
- Tiempo de entrenamiento: LR << RF << XGB
- Proponer: empezar con LR (baseline), luego RF si mejor

---

### Q8: Class Imbalance
**Dificultad:** ⭐⭐⭐  
**Tema:** Data Processing

*"Tu dataset tiene 95% clase negativa, 5% positiva. ¿Cómo entrenas?"*

**Expected:**
- Problema: accuracy es engañosa (95% prediciendo siempre 0)
- Soluciones:
  - Class weights en el modelo
  - Oversampling (SMOTE)
  - Undersampling
  - Stratified split
- Métrica: F1, AUC-ROC, no accuracy

---

### Q9: Feature Importance
**Dificultad:** ⭐⭐  
**Tema:** Feature Engineering

*"Entrenaste un Random Forest. El cliente pregunta: ¿Cuáles son las features más importantes? ¿Cómo respondes?"*

**Expected:**
- RF da importancia basada en ganancia de Gini
- Mostrar top 10 features
- Advertencia: importancia no es causalidad
- Alternativa: SHAP values para explicabilidad más profunda
- Visualizar (barplot, etc.)

---

### Q10: Cross-Validation Strategy
**Dificultad:** ⭐⭐⭐  
**Tema:** ML Fundamentals

*"¿Qué tipo de cross-validation usarías para datos de series de tiempo? ¿Por qué no k-fold normal?"*

**Expected:**
- K-fold normal rompe el orden temporal (data leakage)
- Time series: forward-chaining, expanding window
- Ejemplo: split 80/20 temporal, validate en futuro
- Prevent lookahead bias

---

### Q11: Missing Data Handling (EPAM Official)
**Dificultad:** ⭐⭐⭐  
**Tema:** Data Processing

*"What steps would you take to handle missing data?"*

**Expected:**
- Entender patrón (MCAR vs MAR vs MNAR)
- Opción 1: Eliminar (si <5%)
- Opción 2: Imputación simple (media/mediana)
- Opción 3: Imputación avanzada (KNN, MICE)
- Opción 4: Crear feature "missing indicator"
- Decision tree: depende de % y patrón

---

### Q12: Handling Missing Data in Time Series (EPAM Official)
**Dificultad:** ⭐⭐⭐  
**Tema:** Time Series

*"How would you handle missing data in time series?"*

**Expected:**
- Forward fill / backward fill (datos temporales)
- Interpolación (lineal, polinomial)
- Imputación basada en períodos anteriores
- Validar que no rompe estructura temporal
- Alternativa: dropna si gaps pequeños

---

### Q13: K-Means vs DBSCAN (EPAM Official)
**Dificultad:** ⭐⭐⭐  
**Tema:** Clustering

*"What is the difference between K-Means and DBSCAN?"*

**Expected:**
- K-Means: centroid-based, k clusters, datos esféricos
- DBSCAN: density-based, clusters arbitrarios, eps/minPts
- K-Means: requiere k a priori, DBSCAN: eps/minPts
- K-Means: velocidad, DBSCAN: shapes arbitrarios
- Escalabilidad, sensibilidad a outliers

---

### Q14: Clustering Evaluation (EPAM Official)
**Dificultad:** ⭐⭐⭐  
**Tema:** Clustering

*"Describe how to evaluate a clustering model."*

**Expected:**
- Sin labels: Silhouette, Davies-Bouldin, Calinski-Harabasz
- Con labels: Purity, Rand Index, NMI
- Visualización: t-SNE, UMAP
- Interpretación business

---

### Q15: Clustering for Business Problem (EPAM Official)
**Dificultad:** ⭐⭐⭐  
**Tema:** Clustering

*"Describe a project where you used clustering to solve a business problem."*

**Expected:**
- Problema real definido
- Datos: origen, limpieza
- Preprocesamiento: scaling, features
- Algoritmo elegido y por qué
- Resultados: número clusters, tamaños
- Impacto business: segmentación, decisiones
- Learnings

---

## 🔥 ROUND 3: Specializations (Lunes 10:00-18:00) — Q16-Q30

### Q16: Neural Networks Basics
**Dificultad:** ⭐⭐⭐  
**Tema:** Deep Learning

*"Explica qué hace una neurona artificial. ¿Qué es la función de activación? ¿Por qué no usar activación lineal en todas las capas?"*

**Expected:**
- Neurona: sum(weights * inputs) + bias → activación
- Activación: introduce no-linealidad
- Lineal en todas: red se colapsa a mapeo lineal, sin poder expresivo
- ReLU, Sigmoid, Tanh: cuándo usar cada una
- Output layer: sigmoid para binaria, softmax para multi-clase

---

### Q17: Backpropagation Intuición
**Dificultad:** ⭐⭐⭐⭐  
**Tema:** Deep Learning

*"En 2 minutos: ¿Cómo funciona backpropagation? No necesito las ecuaciones, pero dame la intuición."*

**Expected:**
- Forward pass: predice
- Calcula loss (diferencia predicción vs real)
- Backward pass: propaga el error hacia atrás
- Cada peso ajusta: gradiente del loss respecto a ese peso
- Repetir hasta convergencia

---

### Q18: Convolutional Layers (EPAM Official)
**Dificultad:** ⭐⭐⭐  
**Tema:** Computer Vision

*"How do convolutional layers work in CNNs?"*

**Expected:**
- Filtro desliza sobre imagen
- Detecta features locales (bordes, texturas)
- Neuronas compartidas (eficiencia)
- Fully connected: cada pixel → neurona (explosión de parámetros)
- Ejemplo: primer filtro detecta bordes, segundo combina

---

### Q19: YOLO vs RCNN (EPAM Official)
**Dificultad:** ⭐⭐⭐⭐  
**Tema:** Computer Vision

*"What are the differences between YOLO and RCNN?"*

**Expected:**
- YOLO:
  - Región única, end-to-end
  - Muy rápido (real-time)
  - Menos accuracy
  - Una sola pasada
- R-CNN / Faster R-CNN:
  - Múltiples regiones (RPN)
  - Más lento
  - Mejor accuracy
  - Dos fases
- Trade-off: speed vs accuracy

---

### Q20: Transfer Learning (EPAM Official - Pre-trained vs From Scratch)
**Dificultad:** ⭐⭐⭐  
**Tema:** Computer Vision

*"Describe the trade-offs between using a pre-trained model versus building one from scratch."*

**Expected:**
- Pre-trained:
  - ✓ Transferencia de features (ImageNet, etc.)
  - ✓ Rápido, menos datos
  - ✗ Fine-tuning necesario, ajuste controlado
  - ✓ SOTA performance
- From scratch:
  - ✓ Full control, arquitectura custom
  - ✗ Requiere MUCHOS datos
  - ✗ Training lento, costoso
  - ✓ Si dominio muy específico
- Recomendación: pre-trained + fine-tuning casi siempre

---

### Q21: Image Segmentation Medical (EPAM Official)
**Dificultad:** ⭐⭐⭐⭐  
**Tema:** Computer Vision

*"How would you approach image segmentation for medical imaging?"*

**Expected:**
- Tipos: semantic vs instance
- Arquitectura: U-Net (encoder-decoder)
- Data augmentation: domain-specific (rotations controladas)
- Class imbalance: Dice loss, focal loss
- Evaluation: Dice coefficient, IoU, Hausdorff distance
- Interpretabilidad: vital en medicina
- Validación: radiologists, etc.

---

### Q22: Text Preprocessing (EPAM Official)
**Dificultad:** ⭐⭐⭐  
**Tema:** NLP

*"What preprocessing steps would you apply to text data?"*

**Expected:**
- Tokenización (split en palabras/subwords)
- Lowercasing (normalización)
- Stopwords removal (the, is, a)
- Stemming vs lemmatización (run/running/ran → run)
- Special characters, números
- HTML tags, emojis
- Encoding (UTF-8)
- Order: depende del problema

---

### Q23: Sentiment Analysis Model (EPAM Official)
**Dificultad:** ⭐⭐⭐  
**Tema:** NLP

*"How would you build a sentiment analysis model?"*

**Expected:**
- Datos: labeling, balance (positivo/negativo/neutral)
- Preprocessing: texto limpio
- Features: TF-IDF, embeddings (Word2Vec, BERT)
- Modelo: Logistic Reg → LSTM → Transformer (BERT)
- Evaluation: Accuracy, F1, confusion matrix
- Deployment: inference rápido

---

### Q24: Embeddings & Word2Vec (Original)
**Dificultad:** ⭐⭐⭐  
**Tema:** NLP

*"¿Qué es un embedding? ¿Cómo Word2Vec aprende que 'king' - 'man' + 'woman' ≈ 'queen'?"*

**Expected:**
- Embedding: representación densa en espacio vectorial
- Palabras similares están cercanas
- Word2Vec: Skip-gram o CBOW
- Aprende contexto: palabras que aparecen juntas en textos
- Relaciones algebraicas emergen naturalmente

---

### Q25: Transformers vs RNNs (EPAM Official)
**Dificultad:** ⭐⭐⭐⭐  
**Tema:** NLP

*"What are the advantages of Transformer-based models over RNNs?"*

**Expected:**
- RNN: procesa secuencial, lento, pierde contexto lejano, vanishing gradient
- Transformers: paralelizable, todo a la vez, SOTA
- Atención: calcula relevancia dinámicamente
- Escala mejor con datos grandes
- BERT, GPT, T5 son transformers

---

### Q26: Attention Mechanism (EPAM Official)
**Dificultad:** ⭐⭐⭐⭐  
**Tema:** NLP

*"How do attention mechanisms work in NLP?"*

**Expected:**
- Problema: RNN olvida contexto lejano
- Atención: "qué partes del texto son relevantes para cada palabra"
- Query, Key, Value
- Scaled dot-product attention
- Multi-head attention
- Self-attention

---

### Q27: ARIMA Forecasting (EPAM Official)
**Dificultad:** ⭐⭐⭐⭐  
**Tema:** Time Series

*"Describe the steps to forecast future values using ARIMA."*

**Expected:**
- ARIMA(p,d,q):
  - p: AR (autoregressive)
  - d: I (integrated/diferencing para stationarity)
  - q: MA (moving average)
- Pasos:
  1. Visualizar serie
  2. Test stationarity (ADF test)
  3. Diferenciar si no-estacionario
  4. ACF/PACF plots → elegir p,q
  5. Fit modelo
  6. Diagnósticos
  7. Forecast
- Validar en test set

---

### Q28: Time Series Features (EPAM Official)
**Dificultad:** ⭐⭐⭐  
**Tema:** Time Series

*"What features would you create for a time series forecasting task?"*

**Expected:**
- Lags: X(t-1), X(t-2), ..., X(t-k)
- Rolling statistics: media, std dev últimos n períodos
- Seasonal: X(t-12) para datos mensuales
- Trend: lineal, polinomial
- Calendar: día de semana, mes, festivo
- External: temperatura, competencia, etc.
- Diferenciación: cambios vs niveles

---

### Q29: Seasonality in Time Series (EPAM Official)
**Dificultad:** ⭐⭐⭐  
**Tema:** Time Series

*"How do you identify and address seasonality in time series data?"*

**Expected:**
- Identificar: gráfico, ACF plot, seasonal decompose
- Seasonal subseries plot
- Direcciones:
  1. Diferenciación seasonal (X(t) - X(t-S))
  2. SARIMA (seasonal ARIMA)
  3. Prophet (seasonal components)
  4. Features seasonal explicit (dummy variables)
- Validar: remove seasonality → white noise

---

### Q30: Recommendation System for Streaming (EPAM Official)
**Dificultad:** ⭐⭐⭐⭐  
**Tema:** Recommendation Systems

*"How would you design a recommendation system for a streaming service?"*

**Expected:**
- Cold-start: new users/items
- Types: collaborative, content-based, hybrid
- Features: user preferences, item metadata
- Algorithms: matrix factorization, deep learning, contextual
- Real-time: latency crítica
- A/B testing: CTR, watch-time, retention
- Metrics: coverage, diversity, serendipity (no solo popular)

---

## 🚀 ROUND 4: Advanced Scenarios (Martes 11:00-15:00) — Q31-Q40

### Q31: End-to-End ML Pipeline
**Dificultad:** ⭐⭐⭐⭐  
**Tema:** Full Workflow

*"Describes un proyecto completo que hiciste: datos → modelo → evaluación → deployment. ¿Qué errores cometiste? ¿Qué harías distinto?"*

**Expected:**
- Estructura clara: problem statement → EDA → preprocessing → modeling → evaluation → deployment
- Herramientas reales (Pandas, Scikit-learn, etc.)
- Error real (no inventado): data leakage, bad validation, misunderstanding
- Cómo lo resolviste
- Learnings

---

### Q32: Data Leakage
**Dificultad:** ⭐⭐⭐⭐  
**Tema:** ML Mistakes

*"¿Qué es data leakage? Dame ejemplos. ¿Cómo la detectas?"*

**Expected:**
- Información del futuro filtra al train
- Ejemplos: scale antes de split, target encoding sin CV, time series lookahead
- Detectar: test >> validation performance
- Prevenir: pipeline limpio, temporal integrity

---

### Q33: Model Monitoring
**Dificultad:** ⭐⭐⭐  
**Tema:** Production / MLOps

*"Tu modelo está en producción. ¿Cómo sabes si está degradándose? ¿Qué monitoreas?"*

**Expected:**
- Prediction distribution cambio
- Feature distribution cambio
- Model performance (si labels delayed)
- Concept drift
- Alertas: thresholds
- Retraining strategy

---

### Q34: Interpreting Black Box
**Dificultad:** ⭐⭐⭐⭐  
**Tema:** Explainability

*"Entrenas un XGBoost. Cliente pregunta: ¿Por qué rechazaste mi crédito? ¿Cómo lo explicas?"*

**Expected:**
- SHAP values: descomposición de predicción
- LIME: aproxima modelo localmente
- Feature importance: qué features movieron la aguja
- Explicación clara: no técnica

---

### Q35: Balancing Speed vs Accuracy
**Dificultad:** ⭐⭐⭐  
**Tema:** Trade-offs

*"Cliente necesita predicciones en tiempo real. Pero XGBoost es lento. ¿Entrenarías un modelo más rápido aunque menos exacto?"*

**Expected:**
- Depende contexto:
  - Ad placement (ms): sacrifica accuracy
  - Fraude (segundos): balance
  - Análisis offline: max accuracy
- Técnicas: logistic reg, feature selection, destilación, caching

---

### Q36: Collaborative Filtering Pros/Cons (EPAM Official)
**Dificultad:** ⭐⭐⭐  
**Tema:** Recommendation Systems

*"What are the pros and cons of collaborative filtering?"*

**Expected:**
- Pros:
  - No necesita metadata de items
  - Descubre relaciones ocultas
  - Recomendaciones serendípitas
- Cons:
  - Cold-start (nuevos usuarios/items)
  - Sparsity (matriz vacía)
  - Escalabilidad
  - Popular bias
- Variantes: user-based, item-based, matrix factorization

---

### Q37: Evaluating Recommendation Systems (EPAM Official)
**Dificultad:** ⭐⭐⭐⭐  
**Tema:** Recommendation Systems

*"How would you evaluate a recommendation system?"*

**Expected:**
- Offline: Precision@K, Recall@K, NDCG, MAP, AUC
- Online: CTR, conversion, watch-time, retention, diversity
- A/B testing: holdout, control groups
- Coverage: % items recomendados
- Cold-start metrics
- Serendipity vs accuracy trade-off

---

### Q38: Real-Time Recommendations Challenges (EPAM Official)
**Dificultad:** ⭐⭐⭐⭐  
**Tema:** Recommendation Systems

*"What challenges arise in real-time recommendations?"*

**Expected:**
- Latency: <100ms típicamente
- Scalability: millones de users/items
- Freshness: nuevos items, cambios rápidos
- Cold-start: usuarios nuevos
- Personalization vs scalability
- A/B testing con dinámica rápida
- Actualizar modelos in-flight

---

### Q39: Overlapping Clusters (EPAM Official)
**Dificultad:** ⭐⭐⭐  
**Tema:** Clustering

*"How would you handle overlapping clusters in your data?"*

**Expected:**
- Problema: puntos entre dos clusters
- Soluciones:
  - Soft clustering (fuzzy c-means)
  - Probabilistic: Gaussian Mixture Models
  - Dimensionality reduction primero (PCA, t-SNE)
  - Gaussian Mixture Models: soft assignments
  - Probabilistic graphical models
- Validación: silhouette negativa en overlaps

---

### Q40: Model Degradation & Retraining
**Dificultad:** ⭐⭐⭐⭐  
**Tema:** MLOps / Production

*"¿Cuándo reentrenar un modelo? ¿Cómo decides frecuencia? ¿Cómo monitoreas degradación?"*

**Expected:**
- Concept drift: distribución cambio
- Data drift: input features cambian
- Performance degradation: accuracy baja
- Monitoring: alertas automáticas
- Retraining frequency: diario, semanal, triggered
- Validation: modelo nuevo > modelo viejo
- Rollback strategy: si algo falla

---

## 🎯 Estructura de Respuestas

Cuando respondas cada pregunta en chat:

```
**Q[N]: [Título]**

**Mi respuesta:**
[Tu respuesta aquí — 2-3 párrafos conciso]

**Conceptos clave:**
- Punto 1
- Punto 2
- Punto 3

**Ejemplo/código (si aplica):**
[Snippet corto]
```

---

## 📅 Schedule de Respuestas

| Round | Preguntas | Día | Horario |
|-------|-----------|-----|---------|
| 1 | Q1-Q5 | Jueves | 21:00-22:00 |
| 2 | Q6-Q15 | Viernes | 14:00-15:00 |
| 3 | Q16-Q30 | Lunes | 17:00-18:00 |
| 4 | Q31-Q40 | Martes | 12:45-15:00 |

---

## 📝 Próximos Pasos

1. **Descarga este archivo**
2. **Responde en chat por rounds**
3. **Compila todas las respuestas en `03_mock_interviews/answered_responses.md`**
4. **Repasa 1 día antes de EPAM**

---

**¿Listo? Empezamos Jueves 21:00 con Q1-Q5.**
