# Specialization 1: Deep Learning & Neural Networks

## 🎯 Objetivos
- Entender redes neuronales y arquitecturas
- Forward pass, backpropagation
- Capas comunes: Dense, Conv, RNN
- Optimización y regularización

**Tiempo de estudio:** 45 min

---

## 1.1 Perceptrón Simple

### Historia
El perceptrón (1958) es la unidad básica de redes neuronales.

```
Input: x₁, x₂, ..., xₙ
Pesos: w₁, w₂, ..., wₙ
Bias: b

Output = activation(Σ(wᵢ * xᵢ) + b)
```

### Función de Activación
Sin activación (lineal), la red colapsa a un mapeo lineal:
```
f(x) = Linear(Linear(x)) = Linear(x)  ← sin poder expresivo
```

**Activaciones comunes:**

| Función | Fórmula | Rango | Cuándo |
|---------|---------|-------|--------|
| ReLU | max(0, x) | [0, ∞) | Capas ocultas (standard) |
| Sigmoid | 1/(1+e^(-x)) | (0, 1) | Output binaria |
| Tanh | (e^x - e^(-x))/(e^x + e^(-x)) | (-1, 1) | Ocasiones |
| Softmax | e^xi / Σ(e^xj) | (0, 1), suma=1 | Output multi-clase |

---

## 1.2 Red Neuronal Multicapa

### Arquitectura
```
Input Layer (784 features, ej: MNIST)
    ↓
Hidden Layer 1 (128 neuronas, ReLU)
    ↓
Hidden Layer 2 (64 neuronas, ReLU)
    ↓
Hidden Layer 3 (32 neuronas, ReLU)
    ↓
Output Layer (10 neuronas, Softmax para 10 dígitos)
```

### Forward Pass
```
z₁ = W₁·x + b₁           (combinación lineal)
a₁ = ReLU(z₁)            (activación)
z₂ = W₂·a₁ + b₂
a₂ = ReLU(z₂)
...
ẑ = Wₙ·aₙ₋₁ + bₙ         (output)
ŷ = Softmax(ẑ)           (probabilidades)
```

### Loss (Cross-entropy para clasificación)
```
Loss = -Σ(yᵢ * log(ŷᵢ))

Ejemplo:
y_true = [0, 0, 1, 0, ...]  (clase 2)
y_pred = [0.1, 0.2, 0.6, 0.1, ...]
Loss = -log(0.6) ≈ 0.51

Si y_pred = [0.1, 0.2, 0.01, 0.1, ...] (mal)
Loss = -log(0.01) ≈ 4.6  (penalización alta)
```

---

## 1.3 Backpropagation

### Intuición
1. **Forward pass:** predicción + cálculo de loss
2. **Backward pass:** propagación del error hacia atrás
3. **Update:** ajuste de pesos en dirección opuesta al gradiente

### Ejemplo simple (1 capa)

```
Input: x = 2
Weight: w = 3
Bias: b = 1
Target: y = 10

Forward:
z = w*x + b = 3*2 + 1 = 7
Loss = (z - y)² = (7 - 10)² = 9

Backward:
∂Loss/∂z = 2*(z - y) = 2*(-3) = -6
∂z/∂w = x = 2
∂Loss/∂w = ∂Loss/∂z * ∂z/∂w = -6 * 2 = -12

Update (learning_rate = 0.01):
w_new = w - lr * ∂Loss/∂w = 3 - 0.01*(-12) = 3.12

Repetir muchas veces → w converge a valor óptimo
```

### Chain Rule (Backprop en múltiples capas)
```
∂Loss/∂w₁ = ∂Loss/∂z₃ * ∂z₃/∂a₂ * ∂a₂/∂z₂ * ∂z₂/∂a₁ * ∂a₁/∂z₁ * ∂z₁/∂w₁

(Aplicar chain rule en reversa, de output hacia input)
```

---

## 1.4 Arquitecturas Comunes

### Fully Connected (Dense)
```
Cada neurona conectada a todas las del siguiente nivel
Parámetros: O(n²) (crece cuadrático)
Uso: tabular data, output layers
```

### Convolucional (CNN)
```
Filtros pequeños que se deslizan sobre la imagen
Parámetros: O(n) (mucho menor)
Uso: imágenes, texto
```

### Recurrente (RNN/LSTM)
```
Conexión temporal: ht = f(xt, ht-1)
Usa: sequences, series de tiempo, NLP
```

---

## 1.5 Regularización

### Overfitting en redes
```
Training loss: 0.1
Validation loss: 2.5  ← Gap grande = overfitting
```

### Dropout
```
Durante entrenamiento: apaga aleatoriamente 50% neuronas
Previene co-adaptación (neuronas "dependientes")

def dropout(x, rate=0.5):
    mask = np.random.binomial(1, 1-rate, x.shape)
    return x * mask / (1 - rate)  # Escala para mantener valor esperado
```

### Batch Normalization
```
Normaliza activaciones entre capas
Beneficios:
- Converge más rápido
- Menos sensible a inicialización
- Efecto regularizador
```

### L1/L2 Regularización
```
Loss_total = Loss_data + λ * Σ(|w|)    # L1
Loss_total = Loss_data + λ * Σ(w²)    # L2

Penaliza pesos grandes
```

### Early Stopping
```
Monitorea validation loss
Detén cuando empieza a subir (no esperes train loss = 0)

epochs = 100
best_loss = ∞
patience = 10
wait = 0

for epoch in range(epochs):
    train()
    val_loss = validate()
    
    if val_loss < best_loss:
        best_loss = val_loss
        wait = 0
        save_model()
    else:
        wait += 1
        if wait > patience:
            break  # Detén aquí
```

---

## 1.6 Optimización

### Gradient Descent Variantes

**Vanilla SGD:**
```
w = w - lr * gradient
```

**Momentum:**
```
velocity = momentum * velocity + gradient
w = w - lr * velocity

Acelera convergencia
```

**Adam (Adaptive Moment Estimation):**
```
Combina momentum + RMSprop
Adapta learning rate por parámetro
Mejor elección: standard en 2024
```

**Learning Rate:**
- Demasiado alto: diverge (loss sube)
- Demasiado bajo: converge lentamente
- Learning rate schedule: disminuir con tiempo

```python
# Learning rate schedule
lr = initial_lr * (0.1 ** (epoch // 10))
# Cada 10 épocas: 10x más pequeño
```

---

## 1.7 Inicialización

### Importancia
Inicialización mala → vanishing/exploding gradients

**He Initialization (ReLU):**
```
w ~ N(0, sqrt(2 / n_in))
```

**Xavier Initialization (Sigmoid/Tanh):**
```
w ~ N(0, sqrt(1 / n_in))
```

---

## 🎓 Resumen de Conceptos

- **Neurona:** Input × Weights + Bias → Activation
- **Red:** Capas apiladas de neuronas
- **Forward:** Predicción
- **Backward:** Cálculo de gradientes
- **Update:** Ajuste de pesos
- **Overfitting:** Regularización + dropout + early stopping
- **Optimización:** Adam típicamente mejor

---

## 📚 Lecturas Complementarias

- "Deep Learning" — Goodfellow, Bengio, Courville
- Fast.ai course: https://fast.ai/
- 3Blue1Brown: "Neural Networks" (YouTube)
