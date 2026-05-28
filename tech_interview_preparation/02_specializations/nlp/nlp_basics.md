# Specialization 3: Natural Language Processing (NLP)

## 🎯 Objetivos
- Tokenización, embeddings
- Word2Vec, GloVe, BERT
- Transformers y atención
- Aplicaciones: clasificación, traducción

**Tiempo:** 45 min

---

## 3.1 Preprocessing

### Tokenización
```python
from nltk.tokenize import word_tokenize

text = "Hello world! How are you?"
tokens = word_tokenize(text)
# ['Hello', 'world', '!', 'How', 'are', 'you', '?']
```

### Normalización
```python
text = text.lower()
text = re.sub(r'[^a-zA-Z\s]', '', text)
# Lowercase + quita puntuación
```

### Stopwords
```python
from nltk.corpus import stopwords

stop = stopwords.words('english')
tokens = [w for w in tokens if w not in stop]
# Quita: "the", "is", "a", "and", etc.
```

### Stemming vs Lemmatización
```python
from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

word = "running"
print(stemmer.stem(word))        # "run" (agresivo)
print(lemmatizer.lemmatize(word))  # "run" (lingüístico)

# Diferencia:
# "better" → stem: "better" → lem: "good"
```

---

## 3.2 Embeddings

### one-hot encoding (old)
```
"cat" → [1, 0, 0, 0, 0]
"dog" → [0, 1, 0, 0, 0]

Problemas:
- Dimensión = vocab size (100k+)
- No captura similitud
```

### Word2Vec (2013)
```
Skip-gram: predice contexto dado palabra
CBOW: predice palabra dado contexto

Entrenamiento:
- "The cat sat on the mat"
- Ventana: palabra ± 2 palabras
- "cat" → contexto: [The, sat, on, the]

Resultado: cada palabra → vector denso (100-d)
"cat" ≈ [0.2, -0.5, 0.1, ...]
"dog" ≈ [0.25, -0.48, 0.12, ...]  (similares)
```

### GloVe
- Combina Word2Vec + matriz de co-ocurrencias
- Mejor en general que Word2Vec puro

### Subword Embeddings
```
BPE (Byte Pair Encoding):
"unbelievable" → "un" + "believ" + "able"

Beneficios:
- Palabras raras → combinación de partes conocidas
- Maneja typos mejor
```

---

## 3.3 Transformers

### Atención (Attention)
```
Query (Q): "¿Qué busco?"
Key (K):   "Qué ofrezco"
Value (V): "Mi contenido"

Attention = softmax(Q·K^T / √d) · V

Resultado: para cada palabra, promedio ponderado
de todas las palabras (ponderado por relevancia)
```

### Multi-Head Attention
```
8 heads en paralelo:
- Head 1: ¿Cuál es el sujeto?
- Head 2: ¿Cuál es el verbo?
- Head 3: ¿Quién hace qué?
...

Concatena resultados
```

### BERT (Bidirectional Encoder Representations)
```
Pre-entrenamiento: predecir palabras mascaradas
"The [MASK] sat on the mat"
BERT predice: "cat"

Transferencia: fine-tune en tus datos
```

### GPT (Generative Pre-trained Transformer)
```
Pre-entrenamiento: predecir siguiente palabra
"The cat sat on"
GPT predice: "the"

Autoregresivo: genera token a token
```

---

## 3.4 NLP Tasks

### Text Classification
```
Input: "This movie was amazing!"
Output: Positive (0.92)

Arquitectura:
Input → BERT embedding → Global Avg/Max Pool → Dense(1) → Sigmoid
```

### Sentiment Analysis
```
Clasificación específica de polarity
Input: review text
Output: Positive/Negative/Neutral

Same as text classification
```

### Named Entity Recognition (NER)
```
Input: "John works at Google"
Output: 
- John: PERSON
- Google: ORG

Etiqueta cada token
```

### Machine Translation
```
Encoder-Decoder (Seq2Seq):
Input: "Hello" (English)
Encoder: procesa input
Context vector: compresión
Decoder: genera output en otro idioma
Output: "Hola" (Spanish)

Con Attention: decoder "ve" todos los tokens del input
```

---

## 3.5 Redes Recurrentes (Context)

### RNN Simple
```
ht = tanh(Uh·ht-1 + Wh·xt + bh)

Problema: vanishing gradient
Información antigua se olvida rápido
```

### LSTM (Long Short-Term Memory)
```
Tres gates:
1. Forget gate: ¿qué olvido?
2. Input gate: ¿qué agrego?
3. Output gate: ¿qué salida?

Mantiene cell state a través de tiempo
Mejor para long-range dependencies
```

### GRU (Gated Recurrent Unit)
```
Versión simplificada de LSTM
2 gates en lugar de 3
Más rápido, similar performance
```

**Nota:** Transformers reemplazaron RNNs (paralelizable, long-range)

---

## 3.6 Aplicaciones Prácticas

### Sentiment Analysis
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love this movie!")
# [{'label': 'POSITIVE', 'score': 0.99}]
```

### Zero-shot Classification
```python
classifier = pipeline("zero-shot-classification")
result = classifier(
    "I have a problem with my iPhone",
    ["electronics", "sports", "travel"]
)
# electronics: 0.98
```

### Question Answering
```python
qa = pipeline("question-answering")
result = qa(
    question="What is my name?",
    context="My name is John and I live in NYC"
)
# John
```

---

## 🎓 Resumen

- **Preprocessing:** tokenizar, normalizar, stopwords
- **Embeddings:** Word2Vec (densidad), BERT (contexto)
- **Transformers:** Atención + paralelizable
- **RNNs:** Legacy, reemplazadas por Transformers
- **Tasks:** clasificación, NER, traducción

