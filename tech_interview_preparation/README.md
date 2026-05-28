# 🎯 EPAM Data Scientist Interview Prep — Sprint 22h

**Fecha:** Jueves 21 - Martes 25 mayo 2026  
**Total:** 22 horas concentradas  
**Formato:** 50% teórico + 30% código + 20% mock interviews  
**Status:** 🚀 IN PROGRESS

---

## 📅 Timeline Exacto

| Día | Horario | Duración | Focus | Módulos |
|-----|---------|----------|-------|---------|
| **JUE 21** | 20:00-22:00 | 2h | Warm-up | Fundamentals 1 + Mock Q1-3 |
| **VIE 22** | 11:00-15:00 | 4h | Core 1 | Fundamentals 2-3 + Ejercicios |
| **LUN 25** | 10:00-18:00 | 8h | Core 2 + Deep | Fundamentals 4 + Specializations + Ejercicios |
| **MAR 26** | 11:00-15:00 | 4h | Integración + Mock Final | Repaso + Mock Q16-20 |

**Micro-schedule por día:**

### Jueves 20:00-22:00 (2h)
- 20:00-20:30: Module 1 — ML Basics (teórico)
- 20:30-21:00: Module 1 — Mini ejercicio
- 21:00-22:00: Mock Interview Q1-3 (3 preguntas)

### Viernes 11:00-15:00 (4h)
- 11:00-11:45: Module 2 — Data Processing (teórico)
- 11:45-12:30: Module 2 — Ejercicio
- 12:30-13:15: Module 3 — Model Development (teórico)
- 13:15-14:00: Module 3 — Ejercicio + Descanso
- 14:00-15:00: Mock Interview Q4-8 (5 preguntas)

### Lunes 10:00-18:00 (8h)
- 10:00-10:45: Module 4 — Evaluation Metrics (teórico)
- 10:45-11:30: Module 4 — Ejercicio
- 11:30-12:30: Specialization 1 — Deep Learning (teórico)
- 12:30-13:15: Specialization 1 — Ejercicio
- 13:15-14:00: **ALMUERZO**
- 14:00-14:45: Specialization 2 — Computer Vision (teórico)
- 14:45-15:30: Specialization 2 — Ejercicio
- 15:30-16:15: Specialization 3 — NLP (teórico)
- 16:15-17:00: Specialization 3 — Ejercicio
- 17:00-18:00: Mock Interview Q9-15 (7 preguntas)

### Martes 11:00-15:00 (4h)
- 11:00-11:30: Specialization 4 — Recommendation Systems (teórico)
- 11:30-12:00: Specialization 4 — Ejercicio
- 12:00-12:45: Repaso rápido (conceptos débiles)
- 12:45-15:00: Mock Interview Q16-20 + Scenarios desafiantes (3 preguntas)

---

## 📚 Módulos de Estudio

### **FUNDAMENTALS (Core 60% del tiempo)**

#### Module 1: ML Basics
- Overfitting vs Underfitting
- Bias-Variance tradeoff
- Train/Validation/Test split
- Cross-validation
- Regularization (L1, L2)
- **Ejercicio:** Implementar validación cruzada manualmente

#### Module 2: Data Processing
- Manejo de valores faltantes
- Outliers y detección
- Scaling y normalización
- Encoding categórico (one-hot, label encoding)
- Feature selection
- **Ejercicio:** Limpiar dataset real con Pandas

#### Module 3: Model Development
- Logistic Regression (math + intuición)
- Decision Trees (splitting, entropy, gini)
- Random Forests (bagging, out-of-bag)
- XGBoost (boosting, gradient)
- SVM (kernel trick, margen)
- **Ejercicio:** Comparar modelos en clasificación binaria

#### Module 4: Evaluation Metrics
- Accuracy, Precision, Recall, F1
- ROC-AUC, Precision-Recall curves
- Confusion Matrix
- Regression: R², MSE, RMSE, MAE
- **Ejercicio:** Calcular métricas manualmente, interpretar

### **SPECIALIZATIONS (30% del tiempo)**

#### Specialization 1: Deep Learning & Neural Networks
- Perceptrón y redes multicapa
- Activación: ReLU, Sigmoid, Tanh
- Backpropagation (intuición)
- Gradient descent, learning rate
- Overfitting en redes: dropout, batch norm
- **Ejercicio:** Clasificación MNIST con TensorFlow/PyTorch

#### Specialization 2: Computer Vision
- Convoluciones: cómo funcionan
- CNNs: arquitectura (Conv → ReLU → Pool → FC)
- Transfer Learning: ResNet, VGG
- Data augmentation
- **Ejercicio:** Clasificar imágenes con modelo pre-entrenado

#### Specialization 3: NLP
- Tokenización, embeddings (Word2Vec, GloVe)
- Transformers: attention mechanism
- BERT, GPT conceptualmente
- Sentiment analysis, text classification
- **Ejercicio:** Fine-tune BERT para clasificación

#### Specialization 4: Recommendation Systems
- Collaborative filtering (user-item matrix)
- Content-based filtering
- Matrix factorization (SVD)
- Hybrid approaches
- **Ejercicio:** Sistema simple CF con Surprise lib

---

## 🎬 Mock Interviews (20 preguntas + scenarios)

**Total: 20 preguntas en 5 rounds**

### Round 1 (Jueves 21:00-22:00, 3 preguntas)
Q1, Q2, Q3 — Warm-up (ML basics)

### Round 2 (Viernes 14:00-15:00, 5 preguntas)
Q4-Q8 — Model selection & evaluation

### Round 3 (Lunes 17:00-18:00, 7 preguntas)
Q9-Q15 — Specializations + scenarios

### Round 4 (Martes 12:45-15:00, 5 preguntas)
Q16-Q20 — Challenging scenarios + design questions

---

## ✅ Checklist

### Antes de empezar (Jueves 20:00)
- [ ] Clonar rama `interview` del repo
- [ ] Jupyter notebook activo
- [ ] Datasets descargados (UCI, Kaggle)
- [ ] Environment Python activado

### Diario
- [ ] Revisar roadmap cada mañana
- [ ] Completar ejercicios (commits diarios)
- [ ] Responder mock questions
- [ ] Notas de conceptos débiles

### Antes de EPAM (Martes 15:00)
- [ ] Todos los módulos completados
- [ ] Mock interviews documentados
- [ ] Revisión de gaps
- [ ] Repos pusheados

---

## 📂 Estructura de archivos

```
tech_interview_preparation/
├── README.md (este archivo)
├── 01_fundamentals/
│   ├── 01_ml_basics.md
│   ├── 01_ml_basics.ipynb
│   ├── 02_data_processing.md
│   ├── 02_data_processing.ipynb
│   ├── 03_model_development.md
│   ├── 03_model_development.ipynb
│   ├── 04_evaluation_metrics.md
│   └── 04_evaluation_metrics.ipynb
├── 02_specializations/
│   ├── deep_learning/
│   │   ├── neural_networks.md
│   │   └── neural_networks.ipynb
│   ├── computer_vision/
│   │   ├── cv_basics.md
│   │   └── cv_basics.ipynb
│   ├── nlp/
│   │   ├── nlp_basics.md
│   │   └── nlp_basics.ipynb
│   └── recommendation_systems/
│       ├── recsys_basics.md
│       └── recsys_basics.ipynb
├── 03_mock_interviews/
│   ├── general_questions.md
│   ├── scenario_based.md
│   └── answered_responses.md (llenarás esto)
├── data/
│   └── .gitkeep
└── utils.py
```

---

## 🎯 Success Criteria

✅ Explicas trade-offs sin dudas (overfitting, bias-variance, precision-recall)  
✅ Codificas un modelo end-to-end en <30 min  
✅ Entiendes redes neuronales más allá de "black box"  
✅ Sabes cuándo usar cada especialización  
✅ Respondes mock interviews sin "ummm" largo  
✅ Defiendes decisiones técnicas con fundamento  

---

## 📖 Referencias Externas

- **Alexey Grigorev's Data Science Interviews:** https://github.com/alexeygrigorev/data-science-interviews
- **Kaggle Learn:** https://kaggle.com/learn
- **StatQuest (YouTube):** Conceptos visuales
- **Fast.ai:** Transfer learning, practical deep learning
- **Hugging Face Course:** NLP moderno

---

**Última actualización:** Mayo 21, 2026  
**Estado:** 🚀 READY TO LAUNCH
