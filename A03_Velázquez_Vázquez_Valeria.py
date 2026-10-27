# Actividad 3 - Regresion lineal multiple
# Prediccion de la calificacion final a partir de horas de estudio y tareas
# realizadas, usando LinearRegression() de Scikit-learn.

# ASIGNATURA
# María Fernanda López Aramburo
# Sección D02
# Calendario 2026 B

# INTEGRANTES
# ICOM - Barrera Hermosillo Elizabeth Kristie - 220916338
# ICOM -Rodríguez Luna Paulina - 223378663
# ICOM - Velázquez Vazquéz Valeria - 224785718

# Fecha de entrega: 8 de septiembre del 2026


# librerias necesarias para el desarrollo del modelo
# NumPy: para el manejo eficiente de arreglos numéricos
# y para operaciones matemáticas vectorizadas (por ejemplo, la generación
# de la malla de puntos que conformará el plano ajustado).
import numpy as np

# Pandas: se emplea para estructurar los datos en un DataFrame,
import pandas as pd

# Matplotlib (pyplot): generar las visualizaciones
# -> gráficas, tanto en dos como en tres dimensiones.
import matplotlib.pyplot as plt

# mplot3d de Matplotlib: se requiere para habilitar
# la proyección tridimensional ('3d') en los ejes de las gráficas.
from mpl_toolkits.mplot3d import Axes3D

# clase LinearRegression del submódulo linear_model de Scikit-learn
# implementar el algoritmo de regresión lineal
# que sera entrenado con los datos disponibles
from sklearn.linear_model import LinearRegression

# train_test_split del submódulo model_selection
# dividir el conjunto de datos original en dos subconjuntos,
# uno de entrenamiento y otro de prueba -> de forma aleatoria y reproducible
from sklearn.model_selection import train_test_split

# mean_squared_error, root_mean_squared_error y r2_score
# del submódulo metrics: con ellas se calculan, respectivamente, el error
# cuadrático medio (MSE), la raíz del error cuadrático medio (RMSE) y el
# coeficiente de determinación (R²).
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score

# paleta de colores
COLOR_VERDE  = "#7BE0A8"   # conjunto de entrenamiento
COLOR_ROSA   = "#FF8FC2"   # conjunto de prueba
COLOR_AZUL   = "#7FB8FF"   #plano ajustado / líneas de referencia
COLOR_MORADO = "#B9A6FF"   # etc

print("Librerías importadas correctamente.")


# Se construye el conjunto de datos original (segun la tabla de la actividad)

# cada LLAVE representa una columna del conjunto de datos y cada VLAOR es la lista de observaciones correspondiente
datos = {
    "horas_estudio":     [5, 20, 5, 15, 10, 8, 12, 18, 6, 14,
                        9, 16, 7, 11, 13, 4, 17, 19, 3, 10,
                        6, 14, 8, 12, 15, 7, 9, 11, 13, 16],
    "tareas_realizadas": [8, 2, 9, 11, 10, 7, 10, 3, 8, 12,
                        9, 5, 6, 11, 8, 10, 4, 1, 12, 7,
                        9, 6, 8, 10, 5, 11, 7, 9, 12, 4],
    "calificacion_real": [10, 30, 20, 25, 15, 18, 22, 28, 12, 24,
                        17, 26, 14, 23, 21, 8, 27, 32, 6, 19,
                        13, 25, 16, 20, 29, 15, 18, 12, 24, 31],
}

# diccionario -> DataFrame de Pandas
# acceder mediante nombres de columna
df = pd.DataFrame(datos)

# primeras filas del DataFrame para verificar que la construcción del conjunto de datos se haya realizado correctamente
print("Dimensión del conjunto de datos (filas, columnas):", df.shape)
print(df.head(10))

# estadísticas descriptivas (media, desviación estándar, mínimo, máximo, cuartiles) de cada una de las variables del conjunto de datos
# Esto se realiza como parte de una exploración inicial
print(df.describe())

# exploracion antes del modelado


# Se separan las variables predictoras (X) de la variable objetivo (y)

# Se seleccionan las columnas 'horas_estudio' y 'tareas_realizadas' como variables independientes
# Scikit-learn espera una matriz de dos dimensiones (m x n) como entrada
X = df[["horas_estudio", "tareas_realizadas"]]

# Se selecciona la columna 'calificacion_real' como variable dependiente (objetivo)
# Esta se maneja como una Serie (vector unidimensional), ya que corresponde a un único valor de salida por observación
y = df["calificacion_real"]

print("Forma de la matriz de variables independientes X:", X.shape)
print("Forma del vector de la variable dependiente y:", y.shape)


# dividir el conjunto de datos en un subconjunto de entrenamiento (70%) y un subconjunto de prueba (30%)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,      # 30% de las observaciones para prueba
    random_state=42      # semilla para que la partición sea reproducible (este es un numero comun para esto)
)

# verificar el número de observaciones asignadas a cada subconjunto
print(f"Observaciones totales:        {X.shape[0]}")
print(f"Observaciones de entrenamiento: {X_train.shape[0]} ({X_train.shape[0]/X.shape[0]:.0%})")
print(f"Observaciones de prueba:        {X_test.shape[0]} ({X_test.shape[0]/X.shape[0]:.0%})")


# instanciar y entrenar el modelo de regresión lineal múltiple


# crear instancia del modelo
modelo = LinearRegression()

# Se entrena (ajusta) el modelo utilizando exclusivamente el conjunto de entrenamiento
# se estiman los parámetros θ0, θ1 y θ2
modelo.fit(X_train, y_train)

print("El modelo ha sido entrenado exitosamente")


# extraer e imprimir los parametros estimados por el modelo.

theta_0 = modelo.intercept_          # intercepto del modelo
theta_1 = modelo.coef_[0]            # peso 'horas_estudio'
theta_2 = modelo.coef_[1]            # peso 'tareas_realizadas'

print(f"Theta 0 (intercepto):                {theta_0:.4f}")
print(f"Theta 1 (peso de 'horas_estudio'):    {theta_1:.4f}")
print(f"Theta 2 (peso de 'tareas_realizadas'): {theta_2:.4f}")


# HECHO CON IA -> ecuacion en latex

signo_t1 = "+" if theta_1 >= 0 else "-"
signo_t2 = "+" if theta_2 >= 0 else "-"

# Se construye la cadena de la ecuación en formato mathtext de Matplotlib
# (delimitada por signos de dólar simples, sintaxis compatible con LaTeX
# básico).
ecuacion_str = (
    r"$\hat{y} = " + f"{theta_0:.4f} {signo_t1} {abs(theta_1):.4f}"
    r"\,x_1 " + f"{signo_t2} {abs(theta_2):.4f}" + r"\,x_2$"
)

# Se imprime también la ecuación en texto plano, como respaldo legible
# independientemente del motor de renderizado gráfico.
print("Ecuación del modelo entrenado (texto plano):")
print(f"y_hat = {theta_0:.4f} {signo_t1} {abs(theta_1):.4f}*x1 {signo_t2} {abs(theta_2):.4f}*x2")


# predicciones del modelo sobre el conjunto de prueba

y_pred = modelo.predict(X_test)

print("Primeras predicciones generadas sobre el conjunto de prueba:")
print(np.round(y_pred, 2))


# plano ajustado por el modelo y sdatos originales (entrenamiento y prueba).

# dos variables predictoras, cubriendo el rango observado en el conjunto de datos original
x1_malla = np.linspace(df["horas_estudio"].min(), df["horas_estudio"].max(), 20)
x2_malla = np.linspace(df["tareas_realizadas"].min(), df["tareas_realizadas"].max(), 20)
X1_grid, X2_grid = np.meshgrid(x1_malla, x2_malla)

# evaluar  la ecuación del modelo (theta0 + theta1*x1 + theta2*x2) sobre cada punto de la malla, generando así la superficie del plano ajustado.
Y_grid = theta_0 + theta_1 * X1_grid + theta_2 * X2_grid


# VISUALIZACIONES: se agrupan las 3 graficas del ejercicio (datos iniciales,
# ecuacion del modelo y datos + plano ajustado) en UNA sola figura, con los
# paneles acomodados de forma estetica entre si (2 graficas 3D arriba, una
# al lado de la otra, y la ecuacion como una franja angosta debajo, centrada).

fig = plt.figure(figsize=(15, 9))
fig.patch.set_facecolor("#FDFBFF")

# se define una cuadricula (grid) de 2 filas x 2 columnas: la fila de arriba
# (mas alta) aloja las dos graficas 3D, y la fila de abajo (mas angosta) aloja
# la franja de la ecuacion, ocupando el ancho completo de la figura.
gs = fig.add_gridspec(2, 2, height_ratios=[5, 1], hspace=0.05, wspace=0.15,
                       left=0.04, right=0.98, top=0.90, bottom=0.05)

fig.suptitle("Regresión lineal múltiple — resumen visual del modelo",
             fontsize=16, color="#5B2C87", fontweight="bold")

# 
# Panel 1 (izquierda, arriba): datos iniciales en 3D, diferenciando con
# colores los conjuntos de entrenamiento y de prueba.
ax1 = fig.add_subplot(gs[0, 0], projection="3d")
ax1.set_facecolor("#FDFBFF")

# verde: observaciones del conjunto de entrenamiento
ax1.scatter(
    X_train["horas_estudio"], X_train["tareas_realizadas"], y_train,
    c=COLOR_VERDE, edgecolor="#2E7D5B", s=65, alpha=0.9, label="Entrenamiento (70%)"
)

# rosa: observaciones del conjunto de prueba.
ax1.scatter(
    X_test["horas_estudio"], X_test["tareas_realizadas"], y_test,
    c=COLOR_ROSA, edgecolor="#B23A78", s=65, alpha=0.9, label="Prueba (30%)"
)

# cosas de la grafica
ax1.set_xlabel("Horas de estudio (x₁)", fontsize=10, labelpad=8, color="#3B1E54")
ax1.set_ylabel("Tareas realizadas (x₂)", fontsize=10, labelpad=8, color="#3B1E54")
ax1.set_zlabel("Calificación real (y)", fontsize=10, labelpad=8, color="#3B1E54")
ax1.set_title("Datos iniciales: entrenamiento vs. prueba", fontsize=12, color="#5B2C87", pad=10)
ax1.legend(loc="upper left", fontsize=8)

# Panel 2 (derecha, arriba): datos originales junto con el plano ajustado
# por el modelo, mas los valores predichos sobre el conjunto de prueba.
ax2 = fig.add_subplot(gs[0, 1], projection="3d")
ax2.set_facecolor("#FDFBFF")

# dibujar el plano ajustado por el modelo
ax2.plot_surface(X1_grid, X2_grid, Y_grid, color=COLOR_MORADO, alpha=0.45, edgecolor="none")

# superponer las observaciones de entrenamiento
ax2.scatter(X_train["horas_estudio"], X_train["tareas_realizadas"], y_train,
        c=COLOR_VERDE, edgecolor="#2E7D5B", s=60, label="Entrenamiento (real)")
ax2.scatter(X_test["horas_estudio"], X_test["tareas_realizadas"], y_test,
        c=COLOR_ROSA, edgecolor="#B23A78", s=60, label="Prueba (real)")

# resaltar valores predichos para el conjunto de prueba
ax2.scatter(X_test["horas_estudio"], X_test["tareas_realizadas"], y_pred,
        c=COLOR_AZUL, edgecolor="#1B4F72", marker="^", s=70, label="Prueba (predicha)")

ax2.set_xlabel("Horas de estudio (x₁)", fontsize=10, labelpad=8, color="#3B1E54")
ax2.set_ylabel("Tareas realizadas (x₂)", fontsize=10, labelpad=8, color="#3B1E54")
ax2.set_zlabel("Calificación (y)", fontsize=10, labelpad=8, color="#3B1E54")
ax2.set_title("Datos originales y plano ajustado por el modelo", fontsize=12, color="#5B2C87", pad=10)
ax2.legend(loc="upper left", fontsize=8)

# Panel 3 (abajo, ocupando ambas columnas): ecuacion del modelo entrenado,
# mostrada como una franja de texto con fondo pastel azul.
ax3 = fig.add_subplot(gs[1, :])
ax3.set_facecolor("#EAF4FF")
ax3.axis("off")
ax3.text(0.02, 0.65, "Ecuación del modelo entrenado:", fontsize=12,
        color="#1B4F72", fontweight="bold", va="center", transform=ax3.transAxes)
ax3.text(0.02, 0.20, ecuacion_str, fontsize=16,
        color="#1B4F72", va="center", transform=ax3.transAxes)

plt.show()


# DATAFRAME:
# horas_estudio, tareas_realizadas, calificacion_real, calificacion_predicha y error (diferencia entre la calificación real y la predicha).

resultados = X_test.copy()
resultados["calificacion_real"] = y_test.values
resultados["calificacion_predicha"] = np.round(y_pred, 2)
resultados["error"] = np.round(resultados["calificacion_real"] - resultados["calificacion_predicha"], 2)

# se reinicia el índice del DataFrame de resultados para una lectura poquito mas clara
resultados = resultados.reset_index(drop=True)

print(resultados)


# métricas de evaluación del modelo sobre el conjunto de prueba.

# error cuadrático medio (MSE): promedio de los errores al cuadrado.
mse = mean_squared_error(y_test, y_pred)

# raíz del error cuadrático medio (RMSE): mismas unidades que 'y'.
rmse = root_mean_squared_error(y_test, y_pred)

# coeficiente de determinación (R²): proporción de varianza explicada.
r2 = r2_score(y_test, y_pred)

print(f"Error cuadrático medio (MSE):        {mse:.4f}")
print(f"Raíz del error cuadrático medio (RMSE): {rmse:.4f}")
print(f"Coeficiente de determinación (R²):    {r2:.4f}")
