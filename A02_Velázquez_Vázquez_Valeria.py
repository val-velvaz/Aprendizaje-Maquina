# Importar librerías
import numpy as np # numpy: manejo de arreglos numéricos
import pandas as pd # pandas: tablas de datos
import matplotlib.pyplot as plt # matplotlib: visualización
from sklearn.linear_model import LinearRegression #scikit-learn: modelo de regresión lineal
from sklearn.model_selection import train_test_split #scikit-learn: ayudante para distribuir datos entre entrenamiento y prueba

# Inicializacion de los arreglos que contienen los datos a tratar, y es el salario y X son los años de experiencia
y = np.array([12,14,17,19,22,25,27,30,32,35,42,44,46,48,51,55])
X = np.array([[i] for i in range(0, len(y))]) # X es un arreglo bidimensional de tamaño (len(y), 1)
x = X.flatten() # Variable ayudante cuando es necesario iterar por X, requiriendo un arreglo unidimensional

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8) # Distribución de los datos entre entrenamiento y prueba

model = LinearRegression().fit(X_train, y_train) # Entrenamiento del modelo

score = model.score(X_test, y_test) # Coeficiente de determinación, que tan bueno es el modelo para predecir los valores de prueba basado en su entrenamiento

alpha = model.coef_[0] #Pendiente de la recta modelada
beta = model.intercept_ #Punto de intercepción de la recta modelada

fig = plt.figure(figsize=(14, 8)) #Inicialización de la figura de matplotlib
ROWS, COLS = 2,5
COLS_FOR_PLOTS = 3

plt.rcParams['text.usetex'] = True #Habilita el renderizado de ecuaciones con LaTeX

# Renderizado de la primera gráfica, que muestra la distribucion entre datos de entrenamiento y de prueba
ax1 = plt.subplot2grid((ROWS, COLS),(0,0), colspan=COLS_FOR_PLOTS)
ax1.set_title("Datos de prueba")
ax1.scatter(X_train, y_train, color="blue", label="Datos de entrenamiento")
ax1.scatter(X_test, y_test, color="orange", label="Datos de prueba")

# Renderizado de la segunda gráfica, que muestra la línea de regresión contra los datos iniciales
ax2 = plt.subplot2grid((ROWS, COLS),(1,0), colspan=COLS_FOR_PLOTS)
ax2.set_title(f"Modelo de predicción $y = {np.round(alpha, decimals=4)}x + {np.round(beta, decimals=4)}, R^2={np.round(score, decimals=6)}$") #Muestra el titulo con el valor de la recta modelada y el coeficiente de determinacion
ax2.scatter(X, y, color="blue", label="Datos")
x_predict = np.linspace(0, len(y), 100)
y_predict = x_predict * alpha + beta
ax2.plot(x_predict, y_predict, color="red",label="Línea de regresión")

# Configurar todos los graficos de la misma forma
for ax in [ax1, ax2]:
    ax.set(xlabel="Años de experiencia", ylabel="Salario real (mil MXN)")
    ax.xaxis.set_ticks(np.arange(min(x),max(x)+1, 1.0))
    ax.yaxis.set_ticks(np.arange(min(y),max(y)+1, 5.0))
    ax.legend()
    ax.grid()

# Creacion del DataFrame para comparar directamente datos reales contra predicciones
df = pd.DataFrame()
df['Experiencia']=x
df['Salario real']=y
df['Salario predicho']=np.round(X*alpha + beta, decimals=2)

# Renderizado de la tabla junto a las otras dos figuras
ax3 = plt.subplot2grid((ROWS, COLS), (0,COLS_FOR_PLOTS), rowspan=2, colspan=COLS - COLS_FOR_PLOTS)
ax3.axis("off")
table = ax3.table(cellText = df.values, rowLabels = df.index, bbox=[0,0,1,1], colLabels=df.columns)
table.auto_set_font_size(False)
table.set_fontsize(12)

# Mostrar la figura
plt.tight_layout()
plt.show()
    