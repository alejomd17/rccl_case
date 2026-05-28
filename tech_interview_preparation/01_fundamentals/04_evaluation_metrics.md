# Module 4: Evaluation Metrics — Evaluación de Modelos

## 🎯 Objetivos del módulo
- Entender Confusion Matrix
- Dominar Precision, Recall, F1
- Interpretar ROC-AUC y PR curves
- Usar métricas apropiadas para regresión
- Saber cuándo usar cada métrica

**Tiempo:** 30 min lectura + 30 min ejercicios

---

## 4.1 Confusion Matrix (Matriz de Confusión)

### Entendimiento básico

Para clasificación binaria, hay 4 casos:

```
                Predicción Positiva  Predicción Negativa
Realidad Positiva   TP (True Positive)     FN (False Negative)
Realidad Negativa   FP (False Positive)    TN (True Negative)
```

**Ejemplo: Detector de fraude**

```
              Fraude Detectado  Fraude No Detectado
Fraude Real           TP: 80                FN: 20
No Fraude            FP: 100               TN: 8800

Matriz:
         Pred Fraude  Pred No Fraude
Real Fraude    80          20
Real No Fraude 100        8800
```

### Cálculo

```python
from sklearn.metrics import confusion_matrix
import numpy as np

y_true = [1, 0, 1, 1, 0, 0, 1]
y_pred = [1, 0, 1, 0, 0, 1, 1]

cm = confusion_matrix(y_true, y_pred)
# [[TN, FP],
#  [FN, TP]]

tn, fp, fn, tp = cm.ravel()
```

---

## 4.2 Clasificación: Precision, Recall, F1

### Precision (Precisión)

```
Precision = TP / (TP + FP)

"De lo que predije positivo, cuánto acerté"

Ejemplo fraude:
Precision = 80 / (80 + 100) = 0.44 = 44%

Interpretación: 44% de las alertas de fraude son VERDADERAS.
                56% son falsas alarmas.
```

**Cuándo importa:**
- Email: ¿Spam es realmente spam?
- Ads: ¿Anuncio es realmente relevante?
- Alert: ¿Alertas son reales?

**Lema:** "No quiero muchas falsas alarmas"

### Recall (Sensibilidad, Exhaustividad)

```
Recall = TP / (TP + FN)

"De lo positivo real, cuánto detecté"

Ejemplo fraude:
Recall = 80 / (80 + 20) = 0.80 = 80%

Interpretación: Detecté 80% de los fraudes reales.
                Se me escapó 20%.
```

**Cuándo importa:**
- Medicina: ¿Detecté la enfermedad?
- Fraude: ¿Detecté el fraude?
- Seguridad: ¿Encontré todas las amenazas?

**Lema:** "No quiero perder positivos reales"

### Visual: Precision vs Recall

```
Ejemplo: Detector de fraude

Predicción muy conservadora (alta Precision, baja Recall):
├─ Solo alerta si 99% seguro
├─ Precision: 99%
├─ Recall: 10%
└─ Impacto: Muchos fraudes sin detectar

Predicción muy agresiva (baja Precision, alta Recall):
├─ Alerta por cualquier cosa
├─ Precision: 30%
├─ Recall: 95%
└─ Impacto: Muchas falsas alarmas
```

### F1-Score (Media Armónica)

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)

Rango: 0 a 1 (1 = perfecto)

Ejemplo:
Precision = 0.44, Recall = 0.80
F1 = 2 * (0.44 * 0.80) / (0.44 + 0.80) = 0.57
```

**Cuándo usar:**
- Quieres balance entre Precision y Recall
- Clases desbalanceadas

### Caso de uso: Precisión vs Recall

```
Detector de Cáncer:
├─ Prioridad: Recall alto (no perder pacientes)
└─ F1: baja Precision acceptable si Recall → 99%

Sistema de Recomendación de Ads:
├─ Prioridad: Precision alta (no aburrir usuario)
└─ F1: Recall bajo acceptable si Precision → 90%
```

### Implementación

```python
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1: {f1:.3f}")
```

---

## 4.3 Accuracy (Exactitud)

```
Accuracy = (TP + TN) / Total

"De todas las predicciones, cuántas acerté"

Ejemplo:
Accuracy = (80 + 8800) / (80 + 20 + 100 + 8800) = 0.988 = 98.8%
```

**⚠️ Peligro: Con clases desbalanceadas, Accuracy engaña**

```
Dataset: 99% No-Fraude, 1% Fraude

Modelo dummy (predice siempre "No Fraude"):
Accuracy = 99%  ← ¡Suena bien!
Recall = 0%     ← Pero detecta 0 fraudes

Por eso con desbalance: usar F1, Precision, Recall, ROC-AUC
```

---

## 4.4 ROC-AUC (Receiver Operating Characteristic)

### ¿Qué es?

Curva que muestra el tradeoff entre Tasa de Positivos Verdaderos (Recall) vs Tasa de Falsos Positivos.

```
True Positive Rate (TPR) = TP / (TP + FN)  ← Recall
False Positive Rate (FPR) = FP / (FP + TN)
```

### Gráfico

```
TPR
  |     ╱────╲  Perfect (AUC=1)
  |    ╱      ╲
  |   ╱        ╲  Good (AUC=0.8)
  |  ╱          ╲
  | ╱            ╲
  |╱______________\  Random (AUC=0.5)
  |________________
  0              1  FPR
```

### AUC (Area Under Curve)

```
AUC = 1.0    → Clasificador perfecto
AUC = 0.8    → Muy bueno
AUC = 0.7    → Bueno
AUC = 0.5    → Random (sin poder predictivo)
AUC < 0.5    → Peor que random
```

### Ventajas

- Invariante a threshold
- Resumen en 1 número
- Funciona bien con desbalance

### Implementación

```python
from sklearn.metrics import roc_auc_score, roc_curve

auc = roc_auc_score(y_true, y_pred_proba)
print(f"ROC-AUC: {auc:.3f}")

# Plotear curva
fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, label=f'ROC-AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

---

## 4.5 Precision-Recall Curve

### Cuándo es mejor que ROC-AUC

Con datos **muy desbalanceados**, PR-AUC es más informativo que ROC-AUC.

```
Ejemplo: 99% Negativo, 1% Positivo

ROC-AUC:
└─ FPR no es muy informativi (hay muchos TN, FP pequeño)

PR-AUC:
└─ Precision es lo que importa (¿cuántos positivos predichos son reales?)
```

### Gráfico

```
Precision
  |     ╱────╲  Good classifier
  |    ╱      ╲
  |   ╱        ╲ Random: horizontal (P = fracc positiva)
  |  ╱          ╲
  |_│____________ Recall
  0              1
```

### Implementación

```python
from sklearn.metrics import precision_recall_curve, auc

precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
pr_auc = auc(recall, precision)

plt.plot(recall, precision, label=f'PR-AUC = {pr_auc:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()
```

---

## 4.6 Regresión: R², MSE, RMSE, MAE

Para problemas donde predices un número continuo.

### R² (Coefficient of Determination)

```
R² = 1 - (SS_res / SS_tot)

SS_res = Σ(y_true - y_pred)²  (suma residuales al cuadrado)
SS_tot = Σ(y_true - y_mean)²  (suma variación total)

Rango: -∞ a 1
  1.0 → Predicción perfecta
  0.5 → Explica 50% varianza
  0.0 → No mejor que predecir media
  <0  → Peor que predecir media
```

**Interpretación:** "¿Qué % de varianza explica el modelo?"

```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")
```

### MSE (Mean Squared Error)

```
MSE = (1/n) * Σ(y_true - y_pred)²

Propiedad: Penaliza errores grandes mucho (cuadrado)
```

### RMSE (Root Mean Squared Error)

```
RMSE = √MSE

Ventaja: Mismas unidades que y_true (más interpretable)

Ejemplo: Si RMSE = $5000, predicción desviación promedio es ±$5000
```

### MAE (Mean Absolute Error)

```
MAE = (1/n) * Σ|y_true - y_pred|

Propiedad: Lineal, no penaliza grandes errores tanto
Ventaja: Interpretable, robusta a outliers
```

### Comparación

```
Predicción vs Real:
y_true = [3, 5, 8]
y_pred = [2, 5, 10]

Errores: [-1, 0, 2]

MAE  = (1 + 0 + 2) / 3 = 1.00
MSE  = (1 + 0 + 4) / 3 = 1.67
RMSE = √1.67 = 1.29

Nota: RMSE > MAE porque penaliza más el error 2
```

### Cuándo usar cada una

```
R²: Cuánta varianza explico → resumen general
RMSE: Error promedio (mismas unidades) → predecir precio
MAE: Error promedio robusto → con outliers
```

### Implementación

```python
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

r2 = r2_score(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)

print(f"R²: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
```

---

## 4.7 Decision: ¿Qué métrica usar?

### Clasificación

```
¿Clases balanceadas?
├─ Sí: Accuracy, F1, ROC-AUC todos ok
└─ No: F1 o PR-AUC (Accuracy engaña)

¿Importa Precision?
├─ Sí (Ej: spam filter): Precision-Recall
└─ No: ROC-AUC

¿Threshold flexible?
├─ Sí: ROC-AUC (threshold-independent)
└─ No: Accuracy, F1
```

### Regresión

```
¿Outliers presentes?
├─ Sí: MAE (robusto)
└─ No: RMSE (penaliza más)

¿Qué importa?
├─ Varianza explicada: R²
├─ Error promedio (unidades originales): RMSE o MAE
└─ Todo: Los tres juntos
```

---

## 4.8 Multi-class Classification

Para >2 clases:

### Estrategias

**One-vs-Rest (OvR)**
```
Clase A vs (B,C,D)
Clase B vs (A,C,D)
Clase C vs (A,B,D)
Clase D vs (A,B,C)

Combina resultados
```

**One-vs-One (OvO)**
```
A vs B
A vs C
A vs D
B vs C
B vs D
C vs D

Vota mayoritario
```

### Métricas

```
Macro: Promedia métrica para cada clase (trata clases igual)
Micro: Calcula métricas globales (clases grandes dominan)
Weighted: Promedia ponderado por soporte (balanza clases)
```

```python
from sklearn.metrics import f1_score

f1_macro = f1_score(y_true, y_pred, average='macro')
f1_micro = f1_score(y_true, y_pred, average='micro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')
```

---

## 🎓 Resumen Rápido

### Clasificación Binaria

| Métrica | Cuándo | Fórmula |
|---------|--------|---------|
| Accuracy | Balanceado | (TP+TN)/Total |
| Precision | Falsas alarmas malas | TP/(TP+FP) |
| Recall | Falsos negativos malos | TP/(TP+FN) |
| F1 | Balance Precision-Recall | 2*P*R/(P+R) |
| ROC-AUC | Threshold flexible | ∫(TPR vs FPR) |
| PR-AUC | Muy desbalanceado | ∫(Precision vs Recall) |

### Regresión

| Métrica | Cuándo | Fórmula |
|---------|--------|---------|
| R² | Varianza explicada | 1 - SS_res/SS_tot |
| RMSE | Error promedio (unidades) | √MSE |
| MAE | Error robusto | Mean(|error|) |

---

## 🛠️ Checklist Evaluación

```
☐ Definir métrica de negocio
☐ Crear baseline
☐ Calcular todas las métricas relevantes
☐ Graficar (confusion matrix, ROC, PR)
☐ Comparar en train vs validation vs test
☐ Interpretar: ¿Qué significa este F1 = 0.72?
☐ Validación cruzada (todos los folds)
☐ Decisión: ¿Listo para producción?
```

---

## 📚 Lecturas Complementarias

- Scikit-learn metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
- AUML: https://developers.google.com/machine-learning/crash-course/classification

---

**Next:** Specializations (Deep Learning, Computer Vision, NLP, Recommendation Systems)
