# 🎬 Scenario-Based Interview Questions — EPAM

**Tipo:** Preguntas basadas en escenarios reales  
**Propósito:** Simular situaciones que encontrarás en el trabajo  
**Formato:** Problema ambiguo → tú defines la estrategia

---

## Scenario 1: Fraud Detection System

**Contexto:**
- Banco quiere detectar fraude en transacciones
- Dataset: 1M transacciones/día, 99.5% legítimas, 0.5% fraude
- Latency: predicción en <100ms
- Cost: falso positivo (bloquear cuenta) ≈ falso negativo (perder dinero)

**Preguntas:**
1. ¿Qué métrica optimizarías (accuracy, precision, recall, F1)?
2. ¿Cómo manejarías el desbalance de clases?
3. ¿Qué features extraerías?
4. ¿Qué modelo elegirías y por qué?
5. ¿Cómo deployearías en producción?
6. ¿Cómo monitorearías degradación?

**Respuesta esperada:**
- Métrica: Recall ≥ 95% (no perder fraudes), Precision razonable
- Desbalance: class weights, SMOTE, threshold adjustment
- Features: transacción anterior, monto vs promedio, hora inusual, país nuevo, etc.
- Modelo: Logistic Reg (baseline) → Random Forest → XGBoost
- Deployment: batch + real-time pipeline, cache para decisiones rápidas
- Monitoreo: distribución de scores, tasa de bloqueos, feedback de usuarios

---

## Scenario 2: E-commerce Recommendation

**Contexto:**
- Startup: 50k usuarios, 500k productos
- Objetivo: aumentar AOV (average order value) y retention
- Challenge: cold-start (30% usuarios nuevos cada mes)
- Data: compras históricas, navegación (timestamps), ratings

**Preguntas:**
1. ¿Qué tipo de sistema: collaborative, content-based, hybrid?
2. ¿Cómo manejarías cold-start?
3. ¿Qué métricas usarías para éxito?
4. ¿Cómo A/B testearías?
5. ¿Cuál sería el primero paso (MVP)?

**Respuesta esperada:**
- MVP: item-based CF (fácil, rápido)
- Cold-start: popular items + content-based
- Métricas offline: Precision@5, Recall@5; Online: CTR, AOV, retention
- A/B test: 80/20 split, 2+ semanas (efecto novedad), significance testing
- Roadmap: CF → hybrid → deep learning

---

## Scenario 3: Churn Prediction

**Contexto:**
- SaaS (software as a service) con 100k usuarios
- Problema: 5% churn mensual, costo de reacquisición = $500
- Objetivo: identificar usuarios en riesgo, intervenir antes que cancelen
- Data: uso (logs), soporte (tickets), billing (MRR)

**Preguntas:**
1. ¿Cómo definirías "en riesgo"?
2. ¿Qué features extraerías?
3. ¿Cómo entrenarías sin data leakage?
4. ¿Cómo priorizarías intervenciones?
5. ¿Cómo medirías ROI?

**Respuesta esperada:**
- "En riesgo": actividad ↓50% vs promedio histórico, o tickets de soporte sobre precio
- Features: login frequency, features used, support tickets, NPS score, upgrade status
- Sin leakage: train en historial, test en período futuro, nunca usar "cancellado" flag
- Priorizar: score × LTV (lifetime value) → intervenir top 20%
- ROI: % retenidos con intervención vs sin → cost per prevented churn

---

## Scenario 4: Price Optimization

**Contexto:**
- E-commerce: 10k SKUs
- Precios: dinámicos por demanda, competencia, inventario
- Objetivo: maximizar revenue manteniendo competitividad
- Constraint: no perder clientes a competencia

**Preguntas:**
1. ¿Cómo modelarías elasticidad precio-demanda?
2. ¿Qué datos necesitas?
3. ¿Cómo A/B testearías?
4. ¿Qué algoritmo usarías (regresión, bandits, RL)?
5. ¿Cómo manejarías competencia?

**Respuesta esperada:**
- Elasticidad: regresión log-log o splines (no lineal)
- Datos: precio histórico, cantidad vendida, precios competencia, búsquedas
- A/B test: 5-10% usuarios con precio experimental, 2+ semanas
- Algoritmo: contextual bandits (explora + explota), RL para optimización compleja
- Competencia: monitoreo en tiempo real, no bajar siempre (diferenciación)

---

## Scenario 5: NLP - Sentiment Analysis at Scale

**Contexto:**
- Red social: 1B mensajes/día
- Objetivo: detectar tóxicos, mejorar safety
- Challenge: múltiples idiomas, sarcasmo, slang
- Latency: <100ms por mensaje

**Preguntas:**
1. ¿Cómo estructurarías el pipeline?
2. ¿Qué modelo (BERT, RoBERTa, etc.)?
3. ¿Cómo manejarías múltiples idiomas?
4. ¿Cómo lidiarías con sarcasmo/contexto?
5. ¿Cómo mediría precisión?

**Respuesta esperada:**
- Pipeline: tokenizar → embed → clasificar → action (remove, flag, silenciar)
- Modelo: BERT fine-tuned + ensemble (BERT + LSTM) para robustez
- Idiomas: mBERT (multilingual) o traducir a inglés
- Sarcasmo: feature engineering (negaciones, signos), más datos de entrenamiento
- Precisión: manual review de false positives, metrics per language, online feedback

---

## Scenario 6: Time Series Forecasting - Inventory

**Contexto:**
- Retail: predecir demanda por producto/tienda
- Challenge: seasonality, promociones, out-of-stock events
- Objetivo: minimizar overstock + stockouts
- 5000 productos × 500 tiendas = 2.5M series

**Preguntas:**
1. ¿Qué features usarías?
2. ¿Un modelo por serie o modelo global?
3. ¿Cómo manejarías promociones?
4. ¿Cuál es tu métrica de éxito?
5. ¿Cómo reentrenarías?

**Respuesta esperada:**
- Features: lags (t-1, t-7, t-365), rolling mean, is_promotion, day_of_week, holiday
- Global model (si datos suficientes) + store/product embeddings, o local si sparse
- Promociones: feature explícita (one-hot), o separate model
- Métrica: MAPE (interpretable), pero balance con stockout cost
- Reentrenamiento: semanal (datos nuevos) + triggered si error > threshold

---

## Scenario 7: Computer Vision - Object Detection

**Contexto:**
- Almacén: automatizar conteo de inventario con cámaras
- Challenge: oclusion, variación de iluminación, muchos objetos pequeños
- Latency: <500ms por imagen
- Accuracy: ≥95% para decisiones automáticas

**Preguntas:**
1. ¿Qué modelo (YOLO, Faster R-CNN, RetinaNet)?
2. ¿Cómo manejarías objetos pequeños?
3. ¿Cómo recolectarías datos de entrenamiento?
4. ¿Cómo manejarías domain shift (nueva cámara)?
5. ¿Cómo deployearías?

**Respuesta esperada:**
- Modelo: RetinaNet o Faster R-CNN (mejor en pequeños), no YOLO
- Pequeños objetos: feature pyramid, augmentation (zoom), multiple scales
- Datos: synthetic (renderings) + real (labeling), mix durante entrenamiento
- Domain shift: fine-tune con nuevos datos de cámara, online learning
- Deploy: servidor GPU, batching para throughput, fallback a manual si confidence baja

---

## Scenario 8: Anomaly Detection

**Contexto:**
- Operaciones: detectar fallas en máquinas industriales
- Data: sensores (temperatura, vibración, presión) cada minuto
- Challenge: poco data de anomalías (0.1% normal, 0.9% falla extrema)
- Objetivo: alertar antes de failure catastrophic

**Preguntas:**
1. ¿Supervised o unsupervised?
2. ¿Qué algoritmos (isolation forest, autoencoders, one-class SVM)?
3. ¿Cómo evitas false positives?
4. ¿Cómo evitas false negatives (nuevos tipos de falla)?
5. ¿Cómo explicas al operador por qué alertas?

**Respuesta esperada:**
- Unsupervised (pocas anomalías etiquetadas) + semi-supervised si posible
- Algoritmo: Isolation Forest (rápido) + Autoencoder (reconstrucción error)
- FP: ajustar threshold, domain knowledge (no alertar si es esperado)
- FN: anomaly workshop (recolectar nuevos tipos), reentrenamiento periódico
- Explicabilidad: mostrar sensor que desvía, comparar con baseline histórico

---

## Scenario 9: Click-Through Rate Prediction

**Contexto:**
- Ad tech: predecir probabilidad que usuario haga click en anuncio
- Dataset: 100B clicks/impressions diarios
- Challenge: class imbalance (0.1% CTR), latency <10ms, online learning
- Objetivo: maximizar revenue (clicks × bid price)

**Preguntas:**
1. ¿Cómo manejarías clase imbalance?
2. ¿Qué modelo (logistic reg, GBM, deep learning)?
3. ¿Cómo manejarías feature engineering en escala?
4. ¿Cómo harías online learning?
5. ¿Cómo evaluarías en production?

**Respuesta esperada:**
- Imbalance: class weights, downsampling train, proper metric (AUC, NDCG)
- Modelo: logistic reg (baseline) → XGBoost → deep learning (FM, DeepFM)
- Features: hashing trick (muchos features sparse), feature store
- Online: SGD, mini-batch updates, A/B test nuevas features
- Eval: online metrics (CTR, revenue), calibration checks, compare vs control

---

## Scenario 10: Customer Lifetime Value

**Contexto:**
- Marketing: predecir CLV para priorizar acquisition y retention
- Data: 1M customers, compras históricas, marketing spend, RFM
- Objetivo: decidir presupuesto de marketing por customer
- Constraint: datos incompletos (algunos usuarios jóvenes)

**Preguntas:**
1. ¿Cómo definirías CLV?
2. ¿Regresión o clasificación?
3. ¿Cómo manejarías usuarios nuevos (poco data)?
4. ¿Cómo optimizarías presupuesto?
5. ¿Cómo validarías en tiempo real?

**Respuesta esperada:**
- CLV: suma de margen (compras futuras) - costo de retención estimado
- Regresión (valor continuo), no clasificación, importante es outliers (high CLV)
- Usuarios nuevos: feature engineering de cohorte, transfer learning
- Optimización: presupuesto_marketing ∝ (CLV predicted - retención_costo)
- Validación: predecir CLV passado (validation set), comparar vs real, online feedback

---

## 🎯 Estructura de Respuestas para Scenarios

Para cada scenario, responde:

1. **Problem Understanding** (2 min)
   - ¿Qué entiendes del problema?
   - ¿Cuáles son las métricas de éxito?
   - ¿Cuáles son los constraints?

2. **Approach** (5 min)
   - ¿Qué algoritmo/arquitectura?
   - ¿Por qué ese y no otro?
   - ¿Qué features?

3. **Implementation** (3 min)
   - ¿Cómo lo estructurarías?
   - ¿Qué herramientas/frameworks?
   - ¿Pipeline de datos?

4. **Validation & Deployment** (3 min)
   - ¿Cómo validarías?
   - ¿Cómo deployearías?
   - ¿Cómo monitorearías?

5. **Trade-offs & Improvements** (2 min)
   - ¿Qué trade-offs hiciste?
   - ¿Qué mejorarías después?

---

## 📝 Notas Importantes

- **No hay una respuesta "correcta"** — varía por contexto, datos, recursos
- **Lo importante es el razonamiento** — por qué eliges eso vs otro
- **Menciona trade-offs** — speed vs accuracy, cost vs quality
- **Sé específico** — no "usaría deep learning", sino "BERT fine-tuned porque..."
- **Pregunta hipótesis** — si no tienes información, pregunta al interviewer

---

**Máximo 15 minutos de respuesta por scenario. Sé conciso, decisivo.**

