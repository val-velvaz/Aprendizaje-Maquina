# Actividad 11 - Código k-Means
# Categorización de la base de datos de irises de flores con k-Means de k=3

# ASIGNATURA
# María Fernanda López Aramburo
# Sección D02
# Calendario 2026 B

# INTEGRANTES
# ICOM - Barrera Hermosillo Elizabeth Kristie - 220916338
# ICOM -Rodríguez Luna Paulina - 223378663
# ICOM - Velázquez Vazquéz Valeria - 224785718

# Fecha de entrega: 27 de octubre de 2026

# Importar librerias
import csv  # csv: leer datos iniciales
import math

import matplotlib.lines as mlines  #matplotlib: Leyendas
import matplotlib.pyplot as plt  # matplotlib: Visualización

# abrir/leer archivo.csv
X = []
Y = []
with open("IRIS_PLANT.csv", "r") as cvs_data:
    read_data = csv.DictReader(cvs_data)

    for row in read_data:
        X.append(
            [
                float(row["sepal_length"]),
                float(row["sepal_width"]),
                float(row["petal_length"]),
                float(row["petal_width"]),
            ]
        )
        # Convertir la especie a un valor numérico
        if row["species"] == "Iris-setosa":
            Y.append(0)

        elif row["species"] == "Iris-versicolor":
            Y.append(1)

        elif row["species"] == "Iris-virginica":
            Y.append(2)


class kMeans_Model:
    __k: int
    __cardinality: int

    centroids: list[list[int]]
    __dataset: list[list[int]]

    def __init__(self, k: int, dataset: list[list[int]]):
        self.__dataset = dataset
        self.centroids = []
        self.labels = [] # AGREGADO: Para guardar los grupos finales

        self.__k = k
        self.__cardinality = len(self.__dataset[0])

        # Los centroides son elegidos dividiendo el conjunto en k partes y seleccionando el elemento a la mitad de cada parte
        for i in range(k):
            idx = math.floor(len(self.__dataset) / (2 * k) * (i * 2 + 1))
            self.centroids.append(self.__dataset[idx])

        self.__categorize()

    # Encontrar la distancia eucilidana de dos puntos
    def __calculate_euclidian_for(self, p1, p2):
        if len(p1) != self.__cardinality or len(p2) != self.__cardinality: # CORREGIDO EL ERROR LÓGICO
            raise Exception("Targets are not of the correct cardinality")

        res = 0
        for i in range(self.__cardinality):
            res += (p1[i] - p2[i]) ** 2

        return math.sqrt(res)

    # Calcular las distancias euclidianas de todas las instancias con todos los centroides
    def __calculate_centroid_distances(self):
        distances_per_instace = []
        for p in self.__dataset:
            distances = []
            for c in self.centroids:
                distances.append(self.__calculate_euclidian_for(c, p))
            distances_per_instace.append(distances)

        return distances_per_instace

    # Clasificar cada instancia de acuerdo con el centroide más próximo
    def __classify_instaces(self, distances_per_instance):
        classes = []
        for distances in distances_per_instance:
            min_dist = (-1, math.inf)
            for j, dist in enumerate(distances):
                if dist < min_dist[1]:
                    min_dist = (j, dist)
            classes.append(min_dist[0])

        return classes

    # Promediar cada instancia para cada centroide de acuerdo a su clasificación
    def __interpolate_new_centroids(self, instance_classes):
        self.centroids = [
            [0 for j in range(self.__cardinality)] for i in range(self.__k)
        ]

        tallies = [0 for i in range(self.__k)]
        for i, instance in enumerate(self.__dataset):
            centroid_idx = instance_classes[i]
            tallies[centroid_idx] += 1
            for j, p in enumerate(instance):
                self.centroids[centroid_idx][j] += p

        for i in range(self.__k):
            if tallies[i] > 0: # Evitar división por cero
                self.centroids[i] = [c / tallies[i] for c in self.centroids[i]]

    # Proceso de categorización, acaba cuando los centroides convergen o después de 100 iteraciones
    def __categorize(self):
        for _ in range(100):
            distances_per_instace = self.__calculate_centroid_distances()
            instance_classes = self.__classify_instaces(distances_per_instace)

            prev_centroids = self.centroids.copy()
            self.__interpolate_new_centroids(instance_classes)
            if prev_centroids == self.centroids:
                self.labels = instance_classes # AGREGADO: Guardar las predicciones
                break


kmeans = kMeans_Model(k=3, dataset=X)

# real
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

def data_to_plot(x):
    return [x[0]*x[1], x[2]*x[3]]

def class_to_marker(y):
    marker = ("darkviolet",".")
    if y == 1:
        marker= ("rebeccapurple","*")
    if y == 2:
        marker= ("slateblue","^")
    return marker

# Graficar clases Reales
ax1.set_title("Clases Reales (Dataset Original)")
ax1.set_xlabel("Área del sépalo")
ax1.set_ylabel("Área del pétalo")
for i, instance in enumerate(X):
    marker = class_to_marker(Y[i])
    ax1.plot(*data_to_plot(instance), color=marker[0], marker=marker[1])

# Graficar predicción K-Means
ax2.set_title("Agrupación por K-Means (Predicción)")
ax2.set_xlabel("Área del sépalo")
ax2.set_ylabel("Área del pétalo")
for i, instance in enumerate(X):
    marker = class_to_marker(kmeans.labels[i]) 
    ax2.plot(*data_to_plot(instance), color=marker[0], marker=marker[1])

# Graficar centroides en la gráfica de predicción
for c in kmeans.centroids:
    ax2.plot(*data_to_plot(c), color="orange", marker="1", markersize=12, markeredgewidth=2)

handles = []
for i, label in enumerate(["Iris Setosa", "Iris Versicolor", "Iris Virginica"]):
    marker = class_to_marker(i)
    handles.append(mlines.Line2D([],[],color=marker[0], marker=marker[1], linestyle='None', markersize=10, label=label))
fig.legend(handles=handles, loc='lower center', ncol=3)

#plt.rcParams['text.usetex'] = True # latex
def get_centroid_text(centroids):
    res = ""
    for i,c in enumerate(centroids):
        res += f"C_{i+1}=("
        for j,p in enumerate(c):
            res += f"{p:.2f}"
            if j != len(c)-1: res += ", "
        res += ")   "
    return res

fig.suptitle(get_centroid_text(kmeans.centroids), fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(bottom=0.15, top=0.9)
plt.show()