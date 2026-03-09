# 📊 Proyecto Telecom X (Parte 2) - Predicción de Churn con Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data_Manipulation-green.svg)

## 🎯 Propósito del Proyecto
Este proyecto es la segunda fase del análisis estratégico para la empresa **Telecom X**. El objetivo principal es ir más allá del análisis descriptivo y desarrollar modelos de **Machine Learning** capaces de predecir el *churn* (cancelación o fuga de clientes). A través de la identificación matemática de las variables más relevantes, buscamos clasificar a los usuarios en riesgo y proporcionar a la compañía estrategias proactivas de retención basadas en datos reales.

---

## 📂 Estructura del Proyecto

El repositorio está organizado de la siguiente manera para facilitar su comprensión y reproducibilidad:

```text
📦 TelecomX-Churn-Prediction
 ┣ 📂 data
 ┃ ┗ 📜 TelecomX_Data_Parte2.csv   # Dataset limpio y preprocesado
 ┣ 📂 visualizaciones              # Gráficos generados durante el EDA y modelado
 ┣ 📜 Analisis_TelecomX_Parte2.ipynb # Cuaderno principal con todo el código (EDA + ML)
 ┗ 📜 README.md                    # Documentación del proyecto
```

## ⚙️ Proceso de Preparación de los Datos
Para garantizar que los algoritmos de Machine Learning pudieran interpretar la información correctamente, se realizó un riguroso proceso de Ingeniería de Características (Feature Engineering):

1. Clasificación de Variables:

- Categóricas: Servicios contratados, género, método de pago, tipo de contrato.
- Numéricas: Meses de tenencia (antigüedad), cargos mensuales.

2. Etapas de Normalización y Codificación:

- Binarización: Las variables con respuestas de "Sí/No" (como tener dependientes o ciertos servicios) y el Género se transformaron a valores numéricos 1 y 0.
- One-Hot Encoding (pd.get_dummies): Las variables categóricas complejas como el Tipo de Contrato y Método de Pago se transformaron en múltiples columnas binarias para evitar que el modelo asumiera una jerarquía inexistente.
- Escalado (MinMaxScaler): Exclusivo para modelos basados en distancias (como KNN), donde los datos numéricos fueron normalizados entre 0 y 1 para evitar sesgos por magnitud.

Separación de Datos (Train/Test Split):

- El dataset se dividió usando train_test_split con una proporción del 30% para pruebas (X_test).
- Se aplicó el parámetro stratify=y para asegurar que la proporción de clientes "Activos" y "De baja" se mantuviera idéntica tanto en el conjunto de entrenamiento como en el de validación.


## 🧠 Modelización Predictiva: Justificaciones y Decisiones
Durante la fase de modelado, nos enfrentamos a un problema de Clasificación Desbalanceada (hay muchos más clientes que se quedan que los que se van).

- Métrica de Evaluación: En lugar de optimizar la Exactitud (Accuracy), se priorizó el Recall (Sensibilidad) de la clase minoritaria. En un problema de Churn, es preferible tener "falsos positivos" (ofrecer un descuento a alguien que no se iba a ir) que "falsos negativos" (no detectar a un cliente que termina cancelando).
- Balanceo de Clases: Se implementaron técnicas de Undersampling (NearMiss) y Oversampling (SMOTE). Decisión clave: Estas técnicas se aplicaron únicamente al conjunto de entrenamiento (X_train) utilizando Pipelines para evitar el Data Leakage (fuga de datos) hacia el conjunto de prueba.
- Selección de Modelos: * Se probó un Dummy Classifier como línea base.
   - Se evaluaron Decision Trees y K-Nearest Neighbors (KNN).
   - El modelo final seleccionado fue el Random Forest Classifier, debido a su robustez para manejar múltiples variables categóricas y su capacidad inherente para calcular la "Importancia de las Variables".

## 📈 Insights del Análisis Exploratorio (EDA) e Importancia de Variables
Durante el análisis visual y la posterior extracción de Feature Importances de los modelos, se descubrieron los siguientes factores críticos:

1. ⏱️ La Barrera del Tiempo (Meses de Tenencia): Es la variable más predictiva. El riesgo de fuga es altísimo en los primeros meses. Si un cliente supera el primer año, la probabilidad de retención se dispara.
2. 📄 La Trampa del "Mes a Mes": Los usuarios con contrato Month-to-month representan el grueso de las cancelaciones al no tener barreras de salida.
3. 🛡️ Servicios Ancla: Variables como Soporte Tech y Seguridad Online mostraron un alto impacto positivo. Los clientes que poseen estos servicios rara vez abandonan la compañía.
4. 💰 Sensibilidad al Precio: Los cargos mensuales altos tienen una correlación directa con la cancelación, especialmente si no están respaldados por servicios adicionales.

## 🚀 Instrucciones de Ejecución
Para reproducir este análisis en tu máquina local o en Google Colab:

1. Requisitos previos y librerías
Asegúrate de tener instalado Python 3.8+ y las siguientes bibliotecas. Puedes instalarlas ejecutando:

```Bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn imbalanced-learn
```
2. Carga de datos

- Descarga el archivo TelecomX_Data_Parte2.csv de la carpeta data.
- Abre el archivo Analisis_TelecomX_Parte2.ipynb.
- En la celda correspondiente a la importación de datos (pd.read_csv), actualiza la ruta del archivo para que apunte a donde guardaste el CSV en tu computadora o entorno virtual:

```Python
datos = './ruta/a/tu/archivo/TelecomX_Data_Parte2.csv'
df = pd.read_csv(datos)
```

3. Ejecución
Ejecuta las celdas en orden secuencial. El cuaderno está estructurado lógicamente desde la limpieza de datos hasta la evaluación del último modelo Random Forest.

## ✒️ Autor

**Jesus Antonio Torres Contreras**
* https://www.linkedin.com/in/jesús-antonio-torres-contreras-718069168/
* https://github.com/MrX7Torres

---
*Este proyecto fue realizado como parte de un desafío de análisis de datos enfocado en Machine Learning.*
