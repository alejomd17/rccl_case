# Specialization 2: Computer Vision

## 🎯 Objetivos
- Convoluciones: cómo funcionan
- Arquitecturas: CNN, ResNet, VGG
- Transfer Learning
- Detección de objetos (YOLO, R-CNN)

**Tiempo:** 45 min

---

## 2.1 Convolución

### Concepto
Filtro pequeño (3×3, 5×5) desliza sobre imagen. En cada posición: suma elemento-a-elemento (dot product).

```
Imagen (5×5):          Filtro (3×3):
[1 2 3 4 5]            [0.1 0.2]
[6 7 8 9 0]            [0.3 0.4]
[1 2 3 4 5]
[6 7 8 9 0]
[1 2 3 4 5]

Posición (0,0):
[1 2  ·
 6 7  ·     = 1*0.1 + 2*0.2 + 6*0.3 + 7*0.4 = 3.9
 · · ·]

Resultado: matriz (3×3) de outputs
```

### Parámetros
- **Stride:** cuántos píxeles se mueve el filtro (1, 2, etc.)
- **Padding:** adicionar ceros alrededor (preserva tamaño)
- **Número de filtros:** cuántos se aplican en paralelo

### Interpretación
- Primeras capas: detectan bordes, texturas
- Capas intermedias: patrones (ojos, nariz)
- Capas finales: objetos completos

---

## 2.2 Arquitecturas CNN

### AlexNet (2012)
```
Input (224×224)
  ↓
Conv (96, 11×11) → ReLU → MaxPool
Conv (256, 5×5) → ReLU → MaxPool
Conv (384, 3×3) → ReLU
Conv (384, 3×3) → ReLU
Conv (256, 3×3) → ReLU → MaxPool
  ↓
Flatten → Dense (4096) → Dense (1000)
```

### VGG (2014)
- Bloques de 2-3 Conv pequeñas (3×3) + MaxPool
- Más profunda, parámetros simples
- **VGG16:** 16 capas, 138M parámetros

### ResNet (2015)
```
Innovación: Skip connections

Input
  ↓
Conv → ReLU → Conv
  ↓        ↓
  └─ + ─┘  (suma el input original)
    ↓
  ReLU
```

Beneficios:
- Entrena redes muy profundas (152+ capas)
- Gradientes no se desvanecen

---

## 2.3 Transfer Learning

### Pre-trained Models
Entrenar en ImageNet (1M imágenes, 1000 clases) toma semanas.

**Estrategia:**
1. Descargar modelo pre-entrenado
2. Congelar backbone (no entrenar)
3. Entrenar solo capas finales en tus datos

```python
from torchvision import models

# Cargar ResNet50 pre-entrenado
base_model = models.resnet50(pretrained=True)

# Congelar todos excepto últimas capas
for param in base_model.parameters():
    param.requires_grad = False

# Custom head
base_model.fc = nn.Linear(2048, num_classes)

# Entrenar solo .fc
optimizer = Adam(base_model.fc.parameters(), lr=1e-4)
```

### Cuándo hacer fine-tuning total
Si tienes >100k imágenes propias y dominio muy diferente:
```python
# Descongelar backbone
for param in base_model.parameters():
    param.requires_grad = True

# Learning rate muy bajo (no romper features pre-entrenadas)
optimizer = Adam(base_model.parameters(), lr=1e-5)
```

---

## 2.4 Detección de Objetos

### YOLO (You Only Look Once)
```
Una pasada, una red
Divide imagen en grid (7×7)
Cada celda predice: bounding box + clase + confianza

Ventajas: Rápido (30-60 FPS)
Desventajas: Menos preciso en objetos pequeños
```

### Faster R-CNN
```
Dos fases:
1. RPN (Region Proposal Network): genera ~2000 candidatos
2. Clasificador: clasifica cada región

Ventajas: Preciso, detecta pequeños objetos
Desventajas: Lento (5-10 FPS)
```

---

## 2.5 Segmentación

### Semántica vs Instancia
```
Semántica: clasificar cada píxel
Instancia: separar objetos (contar gatos)
```

### U-Net (encoder-decoder)
```
Encoder (downsample):
Input → Conv → MaxPool → Conv → MaxPool ...

Decoder (upsample):
... → Upsample → Conv → Upsample → Conv → Output

Skip connections: conectan encoder → decoder
```

Uso: segmentación médica, imágenes satélite

---

## 🎓 Resumen

- **Convolución:** filtro + spatial awareness
- **CNN:** capas apiladas de Conv + Pool
- **Transfer Learning:** casi siempre mejor que from scratch
- **YOLO:** velocidad, R-CNN: precisión
- **U-Net:** segmentación

