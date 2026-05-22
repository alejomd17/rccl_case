# Specialization 4: Recommendation Systems

## 🎯 Objetivos
- Collaborative filtering (user-based, item-based, matrix factorization)
- Content-based filtering
- Hybrid approaches
- Evaluación y métricas

**Tiempo:** 45 min

---

## 4.1 Collaborative Filtering

### Intuición
"Usuarios similares ven contenido similar"

```
User-Item Rating Matrix:

        Mov1  Mov2  Mov3  Mov4
User A   5     4     -     2
User B   5     3     4     2
User C   -     4     5     1
User D   4     -     5     3

Tarea: predecir ratings faltantes (-)
```

### User-Based Collaborative Filtering
```
Paso 1: Encontrar usuarios similares a User A
  Similitud(A, B) = cosine(ratings_A, ratings_B)
  
Paso 2: Predecir rating de A en Mov3
  rating_A_Mov3 = promedio_ponderado(ratings_similares[Mov3])
  
Ejemplo:
  B vio Mov3 = 4, similitud(A,B) = 0.9
  C vio Mov3 = 5, similitud(A,C) = 0.7
  rating_A_Mov3 = (4*0.9 + 5*0.7) / (0.9+0.7) ≈ 4.4
```

### Item-Based Collaborative Filtering
```
Alternativa: encontrar items similares

Similitud(Mov1, Mov3) = cosine(ratings_Mov1, ratings_Mov3)

Si User A le gustó Mov1, probablemente le guste Mov3
```

### Matrix Factorization (SVD)
```
Objetivo: factorizar matrix de ratings en dos matrices de menor rango

A ≈ U · Σ · V^T

Dimensiones:
- U: (users × k)      [características latentes de usuarios]
- Σ: (k × k)          [importancia de cada característica]
- V: (items × k)      [características latentes de items]

Ejemplo k=10:
User embedding: [0.2, -0.5, 0.1, ..., 0.3]  (10 dimensiones)
Item embedding: [0.15, -0.48, 0.12, ..., 0.29]  (10 dimensiones)

rating ≈ dot_product(user_embedding, item_embedding)
```

---

## 4.2 Content-Based Filtering

### Concepto
"Te recomiendo items similares a los que te gustaron"

```
Item features: género, actor, director, año, duración
User profile: promedio de características de items que vio

Recomendación: encuentra items similares al profile
```

### Implementación
```python
# Features de películas (one-hot)
item_features = {
    'Inception': [1, 0, 1, 0, 0],  # SciFi, Thriller, ...
    'Interstellar': [1, 0, 1, 1, 0],  # SciFi, Thriller, Drama, ...
    'Comedy1': [0, 1, 0, 0, 1]  # Comedy, ...
}

# User profile = promedio de items que vio
user_profile = [mean de item_features donde rating > 3]

# Similitud con items nuevos
similarities = cosine_similarity(user_profile, all_items_features)
top_recommendations = argsort(similarities)[::-1][:10]
```

---

## 4.3 Cold-Start Problem

### Usuario nuevo
```
Sin histórico → sin similares
Solución:
1. Recomendaciones populares iniciales
2. Preguntar preferencias (onboarding)
3. Hybrid: mezclar popular + exploratorio
```

### Item nuevo
```
Sin ratings → no aparece en recomendaciones
Solución:
1. Content-based (metadatos del item)
2. Exploración: mostrar a usuarios similares a creadores
```

---

## 4.4 Hybrid Approaches

### Combinar múltiples métodos
```python
# Weighted average
score = 0.6 * collab_score + 0.3 * content_score + 0.1 * popularity
```

### Cascade Hybrid
```
1. Content-based: rápido, big set
2. Collaborative: refina top-K
3. Popularity: rellena si falta
```

---

## 4.5 Evaluación

### Offline Metrics
```
Precision@K: % de top-K que usuario realmente vio
Recall@K: % de items que vio que aparecen en top-K
NDCG: normalizado, penaliza ranking malo
MAP: media de precisions en cada K
```

### Online Metrics
```
CTR: % de recomendaciones clickeadas
Conversion: % que compró
Watch-time: minutos vistos
Retention: usuarios que vuelven
```

### Coverage & Diversity
```
Coverage: % items recomendados (no solo top 10)
Diversity: variedad en recomendaciones (no solo blockbusters)
Serendipity: sorpresas positivas

Balance: accuracy vs diversity
```

---

## 4.6 Desafíos Reales

### Sparsity
```
Usuario típico ve 0.001% de items
Matriz 99.9% vacía → patterns débiles
Solución: regularización, SVD
```

### Cold-start
```
Usuarios/items nuevos sin datos históricos
Solución: hybrid + popular + content-based
```

### Scalability
```
M usuarios × M items × tiempo real
Solución: 
- Precompute nearest neighbors
- Índices rápidos (FAISS)
- Cache (Redis)
```

### Popularity Bias
```
Tiende a recomendar solo blockbusters
Nicho content no emergen
Solución: exploration strategy, bandit algorithms
```

---

## 4.7 Aplicaciones Prácticas

### Netflix
```
Billions ratings → matrix factorization + deep learning
Personalization + freshness + diversity
```

### Spotify
```
Collaborative (usuarios similares)
+ Content (características de audio)
+ Context (hora del día, dispositivo)
```

### Amazon
```
Item-based: si viste producto A, te recomendamos B
Pero también: usuarios similares compraron X
```

---

## 🎓 Resumen

- **Collaborative:** usuarios/items similares
- **Content-based:** características explícitas
- **Hybrid:** lo mejor de ambos
- **Cold-start:** reto permanente
- **Métricas:** precisión + cobertura + diversidad

