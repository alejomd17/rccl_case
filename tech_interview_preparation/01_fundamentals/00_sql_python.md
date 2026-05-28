# 🧑‍💻 Round 0: Python & SQL Coding — Questions + Answers

**Tema:** Python (Pandas, NumPy) + SQL (queries, joins, window functions)
**Tiempo:** 60 min
**Formato:** Lee pregunta → responde → compara con respuesta modelo

---

## SQL — Q0.1: Basic Query + Aggregation

**Pregunta:**
*"Tienes una tabla `orders(order_id, customer_id, amount, created_at)`. Escribe una query que retorne el top 5 clientes por total de compras en 2024, incluyendo número de órdenes y monto promedio."*

**Respuesta esperada:**
```sql
SELECT
    customer_id,
    COUNT(order_id)         AS num_orders,
    SUM(amount)             AS total_amount,
    AVG(amount)             AS avg_amount
FROM orders
WHERE YEAR(created_at) = 2024
GROUP BY customer_id
ORDER BY total_amount DESC
LIMIT 5;
```

**Conceptos clave:** GROUP BY, aggregations (COUNT, SUM, AVG), WHERE, ORDER BY, LIMIT

---

## SQL — Q0.2: JOINs

**Pregunta:**
*"Tienes `customers(customer_id, name, country)` y `orders(order_id, customer_id, amount)`. Retorna nombre y país de clientes que han hecho AL MENOS una orden de más de $500. Sin duplicados."*

**Respuesta esperada:**
```sql
SELECT DISTINCT
    c.customer_id,
    c.name,
    c.country
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
WHERE o.amount > 500;
```

**Variante con subquery:**
```sql
SELECT customer_id, name, country
FROM customers
WHERE customer_id IN (
    SELECT DISTINCT customer_id
    FROM orders
    WHERE amount > 500
);
```

**Conceptos clave:** INNER JOIN, DISTINCT, subqueries, ON clause

---

## SQL — Q0.3: Window Functions

**Pregunta:**
*"Tienes `sales(salesperson_id, region, amount, sale_date)`. Para cada vendedor, muestra su monto total de ventas Y el ranking dentro de su región."*

**Respuesta esperada:**
```sql
SELECT
    salesperson_id,
    region,
    SUM(amount) AS total_sales,
    RANK() OVER (
        PARTITION BY region
        ORDER BY SUM(amount) DESC
    ) AS region_rank
FROM sales
GROUP BY salesperson_id, region
ORDER BY region, region_rank;
```

**Conceptos clave:** RANK(), PARTITION BY, OVER(), window functions

---

## SQL — Q0.4: LEFT JOIN + NULL Handling

**Pregunta:**
*"Encuentra todos los clientes que NO han hecho ninguna orden en los últimos 30 días."*

**Respuesta esperada:**
```sql
-- Opción 1: LEFT JOIN + IS NULL
SELECT c.customer_id, c.name
FROM customers c
LEFT JOIN orders o
    ON c.customer_id = o.customer_id
    AND o.created_at >= CURRENT_DATE - INTERVAL 30 DAY
WHERE o.order_id IS NULL;

-- Opción 2: NOT EXISTS
SELECT customer_id, name
FROM customers c
WHERE NOT EXISTS (
    SELECT 1 FROM orders o
    WHERE o.customer_id = c.customer_id
    AND o.created_at >= CURRENT_DATE - INTERVAL 30 DAY
);
```

**Conceptos clave:** LEFT JOIN, IS NULL, NOT EXISTS, INTERVAL

---

## SQL — Q0.5: CTEs + Running Total

**Pregunta:**
*"Calcula el running total (acumulado) de ventas por mes en 2024, ordenado cronológicamente."*

**Respuesta esperada:**
```sql
WITH monthly_sales AS (
    SELECT
        DATE_FORMAT(created_at, '%Y-%m') AS month,
        SUM(amount) AS monthly_total
    FROM orders
    WHERE YEAR(created_at) = 2024
    GROUP BY DATE_FORMAT(created_at, '%Y-%m')
)
SELECT
    month,
    monthly_total,
    SUM(monthly_total) OVER (
        ORDER BY month
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total
FROM monthly_sales
ORDER BY month;
```

**Conceptos clave:** CTE (WITH), SUM() OVER, ROWS BETWEEN, running total

---

## Python — Q0.6: Pandas Basics

**Pregunta:**
*"Tienes un DataFrame de ventas. Encuentra los 3 productos más vendidos por región, con su revenue total. Elimina duplicados y maneja NaNs."*

**Respuesta esperada:**
```python
import pandas as pd

# Setup
df = pd.DataFrame({
    'region': ['North', 'North', 'South', 'South', 'North'],
    'product': ['A', 'B', 'A', 'C', 'A'],
    'revenue': [100, 200, 150, None, 120]
})

# Manejo de NaNs
df['revenue'] = df['revenue'].fillna(df['revenue'].median())

# Top 3 por región
result = (
    df.groupby(['region', 'product'])['revenue']
    .sum()
    .reset_index()
    .sort_values(['region', 'revenue'], ascending=[True, False])
    .groupby('region')
    .head(3)
)

print(result)
```

**Conceptos clave:** groupby, fillna, sort_values, reset_index, head()

---

## Python — Q0.7: NumPy + Estadísticas

**Pregunta:**
*"Dado un array de precios, calcula: media, mediana, desviación estándar, percentil 90, y detecta outliers usando Z-score (|z| > 3)."*

**Respuesta esperada:**
```python
import numpy as np
from scipy import stats

prices = np.array([10, 12, 11, 13, 14, 9, 100, 11, 12, 10])

# Estadísticas
mean = np.mean(prices)
median = np.median(prices)
std = np.std(prices)
p90 = np.percentile(prices, 90)

print(f"Mean: {mean:.2f}")
print(f"Median: {median:.2f}")
print(f"Std: {std:.2f}")
print(f"P90: {p90:.2f}")

# Outlier detection (Z-score)
z_scores = np.abs(stats.zscore(prices))
outliers = prices[z_scores > 3]
print(f"Outliers: {outliers}")

# IQR method
Q1, Q3 = np.percentile(prices, [25, 75])
IQR = Q3 - Q1
outliers_iqr = prices[(prices < Q1 - 1.5*IQR) | (prices > Q3 + 1.5*IQR)]
print(f"Outliers (IQR): {outliers_iqr}")
```

**Conceptos clave:** np.mean/median/std/percentile, Z-score, IQR

---

## Python — Q0.8: Data Cleaning Pipeline

**Pregunta:**
*"Tienes un DataFrame con datos sucios: missing values, duplicados, tipos incorrectos. Escribe un pipeline de limpieza."*

**Respuesta esperada:**
```python
import pandas as pd
import numpy as np

def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete data cleaning pipeline
    """
    print(f"Original shape: {df.shape}")
    
    # 1. Eliminar duplicados
    df = df.drop_duplicates()
    print(f"After dedup: {df.shape}")
    
    # 2. Convertir tipos
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    # 3. Missing values
    missing_pct = df.isnull().mean()
    
    # Columnas con >50% missing: eliminar
    cols_to_drop = missing_pct[missing_pct > 0.5].index
    df = df.drop(columns=cols_to_drop)
    
    # Numéricas: imputar con mediana
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    # Categóricas: imputar con moda
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    
    # 4. Outliers (capping)
    for col in num_cols:
        Q1 = df[col].quantile(0.01)
        Q99 = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=Q1, upper=Q99)
    
    print(f"Final shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    return df
```

**Conceptos clave:** drop_duplicates, pd.to_datetime, fillna, select_dtypes, clip

---

## Python — Q0.9: Feature Engineering

**Pregunta:**
*"Dado un DataFrame de transacciones con columnas `customer_id, date, amount`, crea features para un modelo de churn prediction."*

**Respuesta esperada:**
```python
import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering para churn prediction
    """
    df['date'] = pd.to_datetime(df['date'])
    
    # Referencia temporal
    reference_date = df['date'].max()
    
    features = df.groupby('customer_id').agg(
        # Recency
        days_since_last_purchase=('date', lambda x: (reference_date - x.max()).days),
        
        # Frequency
        num_transactions=('amount', 'count'),
        purchase_frequency_days=('date', lambda x: (x.max() - x.min()).days / max(len(x)-1, 1)),
        
        # Monetary
        total_spend=('amount', 'sum'),
        avg_transaction=('amount', 'mean'),
        std_transaction=('amount', 'std'),
        max_transaction=('amount', 'max'),
        
        # Time-based
        first_purchase=('date', 'min'),
        last_purchase=('date', 'max'),
    ).reset_index()
    
    # Customer lifetime
    features['customer_lifetime_days'] = (
        features['last_purchase'] - features['first_purchase']
    ).dt.days
    
    # Trend: compara últimos 30 días vs total
    recent = df[df['date'] >= reference_date - pd.Timedelta(days=30)]
    recent_spend = recent.groupby('customer_id')['amount'].sum().reset_index()
    recent_spend.columns = ['customer_id', 'recent_30d_spend']
    features = features.merge(recent_spend, on='customer_id', how='left')
    features['recent_30d_spend'] = features['recent_30d_spend'].fillna(0)
    
    return features
```

**Conceptos clave:** groupby + agg, lambda functions, timedelta, merge, RFM features

---

## Python — Q0.10: Sklearn Pipeline

**Pregunta:**
*"Crea un pipeline completo de Scikit-learn para preprocesar datos mixtos (numéricos + categóricos) y entrenar un clasificador. Asegúrate que no haya data leakage."*

**Respuesta esperada:**
```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# Definir columnas
numerical_features = ['age', 'income', 'years_experience']
categorical_features = ['education', 'region', 'job_type']

# Preprocessing numérico
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing categórico
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# Pipeline completo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split ANTES de cualquier preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit y evaluate (sin data leakage)
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')

print(f"Test Accuracy: {accuracy:.4f}")
print(f"CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

**Conceptos clave:** Pipeline, ColumnTransformer, no data leakage, cross_val_score

---

## 📋 Resumen Round 0

| Pregunta | Tema | Concepto Clave |
|----------|------|----------------|
| Q0.1 | SQL | GROUP BY, aggregations, TOP N |
| Q0.2 | SQL | JOINs, DISTINCT, subqueries |
| Q0.3 | SQL | Window functions, RANK, PARTITION BY |
| Q0.4 | SQL | LEFT JOIN + IS NULL, NOT EXISTS |
| Q0.5 | SQL | CTEs, running total, ROWS BETWEEN |
| Q0.6 | Python | Pandas groupby, fillna, top N |
| Q0.7 | Python | NumPy stats, Z-score, IQR |
| Q0.8 | Python | Data cleaning pipeline |
| Q0.9 | Python | Feature engineering, RFM |
| Q0.10 | Python | Sklearn Pipeline, no leakage |

