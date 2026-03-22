import pandas as pd
import datetime
from tkinter import Tk, Label, Button, filedialog
import matplotlib.pyplot as plt

def procesar_archivo():
	file_path = filedialog.askopenfilename(title="Seleccionar archivo CSV", filetypes=[("CSV files", "*.csv")])
	if not file_path:
		return

	try:
		#Carga del dataset
		data = pd.read_csv(file_path)

		#Preparacion dataset
		cols_to_convert = ['Último','Apertura','Máximo', 'Mínimo']
		for col in cols_to_convert:
			data[col] = data[col].str.replace('.','', regex=False).str.replace(',', '.', regex=False).astype(float)

		data['Fecha'] = pd.to_datetime(data['Fecha'], format='%d.%m.%Y')
		data = data.sort_values('Fecha')
		data = data.reset_index(drop=True)

		#Crear las SMA 
		data['SMA_20'] = data['Último'].rolling(window=20).mean()
		data['SMA_100'] = data['Último'].rolling(window=100).mean()


		"""
		Si la SMA de 10 cruza hacia arriba la SMA de 100 y la SMA de 20 también
		está por encima de la SMA de 100 → tendencia alcista → "comprar".

		Si la SMA de 10 cruza hacia abajo la SMA de 100 y la SMA de 20 también
		está por debajo → tendencia bajista → "vender".

		En otros casos → "mantener".
		"""
		#Para considerar si sube o baja, calculando la pendiente "slope"
		prev_sma_20 = data['SMA_20'].iloc[-2]
		prev_sma_100 = data['SMA_100'].iloc[-2]
		current_sma_20 = data['SMA_20'].iloc[-1]
		current_sma_100 = data['SMA_100'].iloc[-1]


		slope_sma_20 = current_sma_20 - prev_sma_20
		slope_sma_100 = current_sma_100 - prev_sma_100

		#Logica de recomendacion
		recomendacion = ""
		if prev_sma_20 < prev_sma_100 and current_sma_20 > current_sma_100:
			recomendacion = "Comprar - SMA 20 cruzó hacia arriba SMA de 100"
		elif prev_sma_20 > prev_sma_100 and current_sma_20 < current_sma_100:
			recomendacion = "Vender - SMA de 20 cruzo hacia abajo la SMA de 100"
		elif slope_sma_20 > 0 and slope_sma_100 > 0:
			recomendacion = "Comprar - posible tendencia alcista"
		else:
			recomendacion= "Mantener - no hay cruce claro"


		#Grafico de las SMA y tendencia de la accion
		plt.figure(figsize=(15,7))
		plt.plot(data['Fecha'], data['Último'], label='Precio Cierre', linewidth=1)
		plt.plot(data['Fecha'], data['SMA_20'], label='SMA a 20 dias', linestyle='--')
		plt.plot(data['Fecha'], data['SMA_100'], label='SMA a 100 dias', linestyle='--')

		plt.text(data['Fecha'].iloc[-1], data['Último'].iloc[-1], recomendacion,
			fontsize=12, color='red', ha='left', va='bottom',
			bbox=dict(facecolor='white', alpha=0.5))

		plt.title('Precio y medias moviles')
		plt.xlabel('Fecha')
		plt.ylabel('Precio')
		plt.legend()
		plt.grid()
		plt.tight_layout()
		plt.show()


	except Exception as e:
		resultado_label.config(text=f"Error: {e}")


#Interfaz
root = Tk()
root.title("Analisis de compra")
root.geometry("400x200")
label = Label(root, text="Selecciona un archivo CSV para analizar", font=('Arial', 12))
label.pack(pady=20)

boton = Button(root, text="Cargar archivo", command=procesar_archivo, font=('Arial', 10), width=20)
boton.pack(pady=10)

resultado_label = Label(root, text="", font=('Arial', 12), fg="blue")
resultado_label.pack(pady=20)

root.mainloop()