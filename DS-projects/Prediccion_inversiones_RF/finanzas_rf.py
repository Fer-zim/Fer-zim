import pandas as pd
from tkinter import Tk, Label, Button, filedialog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix



def centrar_ventana(ventana, ancho=800, alto=400):
	ventana.update_idletasks()
	pantalla_ancho = ventana.winfo_screenwidth()
	pantalla_alto = ventana.winfo_screenheight()
	x = (pantalla_ancho // 2) - (ancho // 2)
	y = (pantalla_alto // 2) - (alto // 2)
	ventana.geometry(f"{ancho}x{alto}+{x}+{y}")



def procesar_archivo():
	file_path = filedialog.askopenfilename(title="Seleccionar archivo CSV", filetypes=[("CSV files", "*.csv")])
	if not file_path:
		return

	try:
		#Carga archivo (CSV)
		data = pd.read_csv(file_path)
		
		#Limpiar columnas, todas reemplazan su coma por punto decimal
		col_convert = ['Último', 'Apertura', 'Máximo', 'Mínimo']
		for col in col_convert:
			data[col] = data[col].str.replace('.','', regex = False).str.replace(',', '.').astype(float)
		#Arreglar fecha
		data['Fecha'] = pd.to_datetime(data['Fecha'], format='%d.%m.%Y')
		data = data.sort_values('Fecha')
		data = data.reset_index(drop=True)

		#Se agrega columna Target, para saber compra=1 venta=0
		data['Target'] = (data['Último'].shift(-1) > data['Último']).astype(int)
		data.dropna(inplace=True)

		#Crear las SMAs 
		data['SMA_20'] = data['Último'].rolling(window=20).mean()
		data['SMA_100'] = data['Último'].rolling(window=100).mean()

		#Elegir variables X y objetivo y
		features = ['SMA_20', 'SMA_100', 'Apertura', 'Máximo', 'Mínimo']
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
		resultado_label.config(text=resultado)

	except Exception as e:
		resultado_label.config(text=f"Error: {e}")

#Interfaz
root = Tk()
root.title("Analisis de compra")

#Centrar ventana
centrar_ventana(root, 400, 200)

#root.geometry("400x200")
label = Label(root, text="Selecciona un archivo CSV para analizar", font=('Arial', 12))
label.pack(pady=20)

boton = Button(root, text="Cargar archivo", command=procesar_archivo, font=('Arial', 10), width=20)
boton.pack(pady=10)

resultado_label = Label(root, text="", font=('Arial', 12), fg="blue")
resultado_label.pack(pady=20)

root.mainloop()
