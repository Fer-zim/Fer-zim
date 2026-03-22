import pandas as pd
import yfinance as yf
import tkinter as tk
from tkinter import ttk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

"""
data = yf.download('AAPL', period='6mo', interval='1d', threads=False)
print(data.head())
"""


def centrar_ventana(ventana, ancho=800, alto=400):
	ventana.update_idletasks()
	pantalla_ancho = ventana.winfo_screenwidth()
	pantalla_alto = ventana.winfo_screenheight()
	x = (pantalla_ancho // 2) - (ancho // 2)
	y = (pantalla_alto // 2) - (alto // 2)
	ventana.geometry(f"{ancho}x{alto}+{x}+{y}")



def obtener_datos():
	ticker = combo.get()
	if not ticker:
		return


	data = yf.download(ticker, period='6mo', interval='1d', threads=False)
	if isinstance(data.columns, pd.MultiIndex):
		data.columns = data.columns.get_level_values(0)


		#Se agrega columna Target, para saber compra=1 venta=0
		data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
		data.dropna(inplace=True)

		#Crear las SMAs 
		data['SMA_20'] = data['Close'].rolling(window=20).mean()
		data['SMA_100'] = data['Close'].rolling(window=100).mean()


		if len(data.dropna()) < 2:
			print("No hay suficientes datos")
			return


		#Elegir variables X y objetivo y
		features = ['SMA_20', 'SMA_100', 'Open', 'High', 'Low']
		X = data[features]
		y = data['Target']

		#Separar el set en train y test
		X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, shuffle=False)

		#Aplicar el modelo
		model = RandomForestClassifier(n_estimators=100, random_state=42)
		model.fit(X_train, y_train)

		y_pred = model.predict(X_test)
		print(confusion_matrix(y_test, y_pred))
		print(classification_report(y_test, y_pred))

		#Prediccion
		next_day = data[features].iloc[[-1]]

		prediction = model.predict(next_day)

		if prediction[0] == 1:
			resultado = "Prediccion: COMPRAR"
		else:
			resultado = "Prediccion: VENDER"

		print(resultado)



#Interfaz
root = tk.Tk()
root.title("Analisis de compra con Random Forest")

#Centrar ventana
centrar_ventana(root, 400, 200)


#Menu desplegable
tk.Label(root, text="Seleccione un Ticker").pack(pady=10)
tickers = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOGL', 'AAL', 'META', 'ALUA.BA', 'AMZN',
			'GLD','YPF']
combo = ttk.Combobox(root, values=tickers, state="readonly")
combo.pack(pady=5)
combo.current(0)

btn = tk.Button(root, text="Obtener", command=obtener_datos)
btn.pack(pady=20)
root.mainloop()

