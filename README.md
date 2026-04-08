# 🔍 Anomaly Detection con Isolation Forest

> Pipeline completo de detección de anomalías aplicable a **fraude financiero**, **sensores IoT** y **análisis de logs**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Descripción

Este proyecto implementa un pipeline end-to-end de detección de anomalías usando **Isolation Forest**, uno de los algoritmos más efectivos para identificar datos atípicos en datasets con alta dimensionalidad.

### ¿Qué incluye?

- ✅ Generación de datos sintéticos (transacciones financieras)
- ✅ Preprocesamiento y feature engineering
- ✅ Entrenamiento con Isolation Forest
- ✅ Evaluación con métricas de clasificación
- ✅ 4 visualizaciones profesionales listas para presentar

---

## 🏗️ Arquitectura del pipeline

```
Data Source → Preprocessing → Model Training → Detection → Evaluation → Visualizations
   (CSV)      (StandardScaler)  (Isolation Forest)  (Scoring)   (Metrics)   (Matplotlib)
```

---

## 🚀 Quickstart

### 1. Clona el repositorio

```bash
git clone https://github.com/tu-usuario/anomaly-detection-python.git
cd anomaly-detection-python
```

### 2. Crea un entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Instala dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecuta el pipeline

```bash
python src/anomaly_detector.py
```

Los gráficos se guardarán automáticamente en la carpeta `outputs/`.

---

## 📊 Resultados

El pipeline genera 4 visualizaciones:

| Gráfico | Descripción |
|---------|-------------|
| `scatter_anomalies.png` | Scatter plot de monto vs frecuencia con anomalías marcadas |
| `anomaly_scores.png` | Distribución de anomaly scores (normal vs anómalo) |
| `confusion_matrix.png` | Matriz de confusión del modelo |
| `time_series.png` | Serie temporal con anomalías detectadas |

---

## 🧠 ¿Cómo funciona Isolation Forest?

Isolation Forest aísla observaciones seleccionando aleatoriamente una feature y un valor de corte. Las anomalías, al ser diferentes de la mayoría, requieren **menos particiones** para ser aisladas.

**Ventajas:**
- No necesita datos etiquetados (aprendizaje no supervisado)
- Escalable a datasets grandes
- Robusto con datos de alta dimensionalidad
- Bajo costo computacional

---

## 📁 Estructura del proyecto

```
anomaly-detection-python/
├── src/
│   └── anomaly_detector.py    # Pipeline completo
├── outputs/                   # Gráficos generados (auto-creado)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔧 Personalización

Puedes adaptar el pipeline a tu caso de uso modificando:

```python
# Cambiar la proporción de anomalías esperadas
model = train_isolation_forest(df, contamination=0.10)

# Usar tus propios datos
df = pd.read_csv("tu_dataset.csv")
```

---

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Abre un issue o envía un PR.

1. Fork el proyecto
2. Crea tu feature branch (`git checkout -b feature/nueva-feature`)
3. Commit tus cambios (`git commit -m 'Agrega nueva feature'`)
4. Push a la branch (`git push origin feature/nueva-feature`)
5. Abre un Pull Request

---

## 📄 Licencia

Distribuido bajo la licencia MIT. Consulta `LICENSE` para más información.

---

## ⭐ ¿Te fue útil?

Dale una estrella al repo y sígueme en [LinkedIn](https://linkedin.com/in/tu-usuario) para más contenido de Data Science y Machine Learning.
