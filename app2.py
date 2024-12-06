# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 08:54:52 2024

@author: emman
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Título de la aplicación
st.title("Sube tu archivo CSV y analiza los datos")

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

    # Paso 4: Estadísticas descriptivas de las columnas numéricas
    st.write("Estadísticas descriptivas de los datos limpios:")
    
    # Seleccionar solo las columnas numéricas
    numeric_columns = df_cleaned.select_dtypes(include=np.number).columns

    # Mostrar estadísticas básicas
    if len(numeric_columns) > 0:
        st.write(df_cleaned[numeric_columns].describe())
        
        # Calcular variables estadísticas adicionales
        for col in numeric_columns:
            mean = df_cleaned[col].mean()
            std_dev = df_cleaned[col].std()
            median = df_cleaned[col].median()
            min_val = df_cleaned[col].min()
            max_val = df_cleaned[col].max()
            q1 = df_cleaned[col].quantile(0.25)
            q3 = df_cleaned[col].quantile(0.75)

            # Mostrar los resultados
            st.write(f"**Estadísticas de la columna '{col}':**")
            st.write(f"  - Media: {mean}")
            st.write(f"  - Desviación estándar: {std_dev}")
            st.write(f"  - Mediana: {median}")
            st.write(f"  - Mínimo: {min_val}")
            st.write(f"  - Máximo: {max_val}")
            st.write(f"  - Cuartil 1 (Q1): {q1}")
            st.write(f"  - Cuartil 3 (Q3): {q3}")
    else:
        st.write("No se encontraron columnas numéricas en el archivo.")

    # Paso 5: Mostrar un gráfico (Ejemplo: gráfico de barras con nombres en el eje X)
    st.write("Generando un gráfico de barras de los Ingresos por Nombre")
    
    # Usar la columna "Nombre" para el eje X y "Ingreso" para el eje Y
    if "Nombre" in df_cleaned.columns and "Ingreso" in df_cleaned.columns:
        plt.figure(figsize=(10, 6))
        plt.bar(df_cleaned["Nombre"], df_cleaned["Ingreso"], color='skyblue')

        # Añadir títulos y etiquetas
        plt.title("Ingresos por Persona")
        plt.xlabel("Nombre")
        plt.ylabel("Ingreso ($)")

        # Mostrar el gráfico
        st.pyplot(plt)

    # Paso 6: Descargar archivo procesado
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
