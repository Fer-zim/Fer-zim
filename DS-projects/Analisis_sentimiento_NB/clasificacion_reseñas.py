#Reseñas
#Naive Bayes - clasificacion de una reseña en negativa o positiva


import pandas as pd

#Se toma el excel
#df = pd.read_excel('C:/Users/1/Desktop/reseñas.xlsx')
#Se convierte el excel en csv
#df_1 = df.to_csv('C:/Users/1/Desktop/reseñas.csv')

#Se lee el csv
df = pd.read_csv('C:/Users/1/Desktop/reseñas.csv')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#2a -  Sin limpiar los caracteres
# 2. Convertir texto a números (Vectores de frecuencias)
# El modelo no entiende palabras, entiende cuántas veces aparece cada una.
#vectorizador = CountVectorizer()
#X = vectorizador.fit_transform(df['Reseña'])

#2b - Limpieza de caracteres
# Agregamos parámetros al vectorizador:
# 1. lowercase=True (convierte todo a minúscula automáticamente)
# 2. stop_words (le pasamos una lista de palabras en español para ignorar)
# 3. token_pattern (usamos una regla para ignorar números y signos)

vectorizador = CountVectorizer(
    lowercase=True,
    ngram_range=(1, 2), # 1=palabras sueltas ("bueno"), 2=parejas ("no bueno")
    stop_words= None, #['el', 'la', 'los', 'las', 'un', 'una', 'de', 'que', 'y', 'en'], 
    token_pattern=r'\b[a-zA-Záéíóúüñ]{2,}\b' # Solo palabras de 3 letras o más, sin números
)

X = vectorizador.fit_transform(df['Reseña'])


# 3. Entrenar el modelo Naive Bayes
modelo = MultinomialNB()
modelo.fit(X, df['Sentimiento'])

# 4. Probemos con un mensaje nuevo
nuevo_mensaje = ["El producto no sirve. es pesima la respuesta que se obtiene, se cuelga y no hay soporte"]
nuevo_X = vectorizador.transform(nuevo_mensaje)

prediccion = modelo.predict(nuevo_X)
probabilidades = modelo.predict_proba(nuevo_X)

# 5. Resultado 
resultado = prediccion[0] # Esto ya devolverá "Positivo" o "Negativo"
print(f"Resultado: {resultado}")

# Para las probabilidades, recuerda el orden alfabético:
# probabilidades[0][0] es "Negativo"
# probabilidades[0][1] es "Positivo"
print(f"Probabilidades: Negativo: {probabilidades[0][0]:.2f}, Positivo: {probabilidades[0][1]:.2f}")
