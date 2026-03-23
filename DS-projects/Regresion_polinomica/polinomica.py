import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



np.random.seed(0)
X = 6 * np.random.rand(100, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)



# Visualizamos los datos
plt.scatter(X, y, color='blue')
plt.xlabel("Variable Independiente X")
plt.ylabel("Variable Dependiente Y")
plt.title("Datos sintéti cos para regresión polinomial")
plt.show()


# Transformación a segundo grado
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)


# Ajuste del modelo de regresión lineal en el espacio transformado
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# Predicciones
y_pred = lin_reg.predict(X_poly)

# Visualización del modelo ajustado
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red', linewidth=2)
plt.xlabel("Variable Independiente X")
plt.ylabel("Variable Dependiente Y")
plt.title("Ajuste de regresión polinomial (grado 2)")
plt.show()


mse = mean_squared_error(y, y_pred)
print(f"Error cuadrático medio (MSE): {mse}")