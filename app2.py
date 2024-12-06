# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 08:54:52 2024

@author: emman
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Título de la aplicación
st.title("Análisis de datos con Streamlit")

# Paso 1: Subir un archivo CSV
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    # Paso 2: Leer el archivo CSV
    df = pd.read_csv(uploaded_file)
    
    # Mostrar los primeros 5 registros
    st.write("Primeros 5 registros del archivo:")
    st.write(df.head())

    # Paso 3: Limpiar los datos (eliminar valores nulos)
    st.write("Limpiando los datos...")
    df_cleaned = df.dropna()

    # Paso 4: Mostrar un gráfico (Ejemplo: gráfico de barras con nombres en el eje X)
    st.write("Generando un gráfico de barras de los Ingresos por Nombre")
    
    # Usar la columna "Nombre" para el eje X y "Ingreso" para el eje Y
    plt.figure(figsize=(10, 6))
    plt.bar(df_cleaned["Nombre"], df_cleaned["Ingreso"], color='skyblue')

    # Añadir títulos y etiquetas
    plt.title("Ingresos por Persona")
    plt.xlabel("Nombre")
    plt.ylabel("Ingreso ($)")

    # Mostrar el gráfico
    st.pyplot(plt)

    # Paso 5: Descargar archivo procesado
    st.write("¿Te gustaría descargar el archivo procesado?")
    
    # Convertir el DataFrame limpio a CSV
    cleaned_csv = df_cleaned.to_csv(index=False)
    
    # Crear un enlace para descargar el archivo
    st.download_button(
        label="Descargar archivo CSV procesado",
        data=cleaned_csv,
        file_name="archivo_procesado.csv",
        mime="text/csv"
    )
