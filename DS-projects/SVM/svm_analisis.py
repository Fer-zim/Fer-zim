import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# 1. Creacion de datos
#desviación estándar un poco alta para que haya solapamiento (cluster_std)
X, y = datasets.make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.5)

# Visualización 1: Los datos originales sin separar
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='winter', edgecolors='k')
plt.title("Visualización de Datos")
plt.xlabel("Variable 1")
plt.ylabel("Variable 2")
plt.grid(True)
plt.show()

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Configuracion y entrenamiento de SVM (clasificador)
#kernel lineal y un parámetro C estándar.
# Un C alto (e.g., 10) intentará un margen más estricto.
# Un C bajo (e.g., 0.1) será más tolerante. Usaremos 1.0 por defecto.
modelo = SVC(kernel='linear', C=1.0)
modelo.fit(X_train, y_train)

# Predicciones
y_pred = modelo.predict(X_test)

def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Grafica la función de decisión para un SVC en 2D."""
    if ax is None: #como no se le pasa un lienzo, se le pasa una ventana del grafico, entocnes toma ese como lienzo y pide sus valores de x e y
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Crear rejilla para evaluar el modelo
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # Dibujar fronteras y márgenes
    # levels=[-1, 0, 1] corresponden a los márgenes y al hiperplano central
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # Resaltar vectores de soporte
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                   s=300, linewidth=1, edgecolors='black', facecolors='none', label='Vectores de Soporte')


# Visualización 2: El modelo aplicado
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='winter', edgecolors='k')
plot_svc_decision_function(modelo)
plt.title("SVM con Kernel Lineal: Hiperplano, Márgenes y Vectores de Soporte")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.legend()
plt.show()

# Evaluación numérica
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))





# 1. Definimos un nuevo punto (coordenadas x e y)
# los valores deben estar dentro del rango de los datos previos
punto_nuevo = np.array([[-2, 12]]) 

# 2. El modelo predice la clase (0 o 1)
prediccion = modelo.predict(punto_nuevo)
clase_nombre = "Azul" if prediccion[0] == 0 else "Verde"

print(f"El punto {punto_nuevo[0]} ha sido clasificado como: {clase_nombre}")

# 3. Visualización para confirmar
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='winter', alpha=0.5) # Datos originales
plot_svc_decision_function(modelo) # Dibujamos la frontera

# Dibujamos el nuevo punto con un marcador especial (una estrella roja)
plt.scatter(punto_nuevo[0, 0], punto_nuevo[0, 1], c='red', marker='*', s=200, label=f'Predicción: {clase_nombre}')

plt.title("Clasificación de un nuevo punto con SVM")
plt.legend()
plt.show()