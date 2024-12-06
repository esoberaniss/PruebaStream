# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 06:48:24 2024

@author: emman
"""

import streamlit as st
import pandas as pd
import numpy as np
import io

# Título de la aplicación
st.title("Sube tu archivo CSV")

# Paso 1: Subir un archivo CSV
uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

# Paso 2: Procesar el archivo una vez subido
if uploaded_file is not None:
    # Leer el archivo CSV usando pandas
    df = pd.read_csv(uploaded_file)
    
    # Mostrar el DataFrame
    st.write("Contenido del archivo:")
    st.write(df)
    
    # Paso 3: Realizar una operación (Ejemplo: Calcular la media de una columna numérica)
    if st.button("Calcular la media de la primera columna numérica"):
        # Filtrar las columnas numéricas
        numeric_columns = df.select_dtypes(include=np.number).columns
        if numeric_columns.size > 0:
            # Calcular la media de la primera columna numérica
            column_name = numeric_columns[0]
            mean_value = df[column_name].mean()
            st.write(f"La media de la columna '{column_name}' es: {mean_value}")
        else:
            st.write("No se encontraron columnas numéricas en el archivo.")

    # Paso 4: Guardar el resultado (opcional, por ejemplo, exportar los datos procesados)
    result_csv = df.to_csv(index=False)
    st.download_button("Descargar archivo procesado", result_csv, "resultados.csv", "text/csv")

