# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 09:19:54 2024

@author: emman
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Título de la aplicación
st.title("Análisis y Predicción de Datos con Streamlit")

# Paso 1: Subir un archivo CSV
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    # Leer el archivo CSV
    df = pd.read_csv(uploaded_file)
    
    # Mostrar los primeros 5 registros
    st.write("Primeros 5 registros del archivo:")
    st.write(df.head())
    
    # Paso 2: Limpieza de Datos
    st.write("Limpieza de datos:")
    
    # Eliminar filas con valores nulos
    df_cleaned = df.dropna()
    
    # Convertir columnas de texto a categóricas (si aplica)
    for col in df_cleaned.select_dtypes(include='object').columns:
        df_cleaned[col] = df_cleaned[col].astype('category')
    
    st.write("Datos limpiados:")
    st.write(df_cleaned.head())
    
    # Paso 3: Estadísticas descriptivas de las columnas numéricas
    st.write("Estadísticas descriptivas de los datos limpios:")
    
    # Seleccionar solo las columnas numéricas
    numeric_columns = df_cleaned.select_dtypes(include=np.number).columns

    if len(numeric_columns) > 0:
        st.write(df_cleaned[numeric_columns].describe())
    else:
        st.write("No se encontraron columnas numéricas en el archivo.")
    
    # Paso 4: Análisis de Correlación
    st.write("Análisis de correlación entre columnas numéricas:")
    correlation_matrix = df_cleaned[numeric_columns].corr()
    st.write(correlation_matrix)
    
    # Mostrar el gráfico de correlación
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt)

    # Paso 5: Visualización interactiva con Plotly
    st.write("Generando gráfico de dispersión interactivo (Edad vs Ingreso):")
    
    # Si las columnas 'Edad' e 'Ingreso' existen, crear un gráfico interactivo de dispersión
    if 'Edad' in df_cleaned.columns and 'Ingreso' in df_cleaned.columns:
        fig = px.scatter(df_cleaned, x="Edad", y="Ingreso", title="Edad vs Ingreso", labels={"Edad": "Edad", "Ingreso": "Ingreso"})
        st.plotly_chart(fig)
    
    # Paso 6: Modelo de Regresión Lineal
    st.write("Entrenando un modelo de regresión lineal para predecir Ingreso basado en Edad:")
    
    if 'Edad' in df_cleaned.columns and 'Ingreso' in df_cleaned.columns:
        # Preparar los datos para el modelo
        X = df_cleaned[['Edad']]  # Usar 'Edad' como predictor
        y = df_cleaned['Ingreso']  # Usar 'Ingreso' como objetivo
        
        # Dividir en conjunto de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Crear el modelo de regresión lineal
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Realizar predicciones
        y_pred = model.predict(X_test)
        
        # Mostrar el error cuadrático medio
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"El error cuadrático medio (MSE) es: {mse}")
        
        # Mostrar la predicción frente a la realidad
        st.write("Predicciones vs Valores reales:")
        prediction_df = pd.DataFrame({"Real": y_test, "Predicción": y_pred})
        st.write(prediction_df.head())
        
        # Graficar las predicciones vs los valores reales
        plt.figure(figsize=(10, 6))
        plt.scatter(X_test, y_test, color='blue', label='Real')
        plt.plot(X_test, y_pred, color='red', label='Predicción')
        plt.title("Regresión Lineal: Ingreso vs Edad")
        plt.xlabel("Edad")
        plt.ylabel("Ingreso")
        plt.legend()
        st.pyplot(plt)

    # Paso 7: Descargar los Resultados Procesados y Predicciones
    st.write("¿Te gustaría descargar el archivo con las predicciones?")
    
    # Crear un DataFrame con las predicciones
    prediction_results = df_cleaned.copy()
    prediction_results['Predicción Ingreso'] = model.predict(df_cleaned[['Edad']])
    
    # Convertir el DataFrame con predicciones a CSV
    result_csv = prediction_results.to_csv(index=False)
    
    # Crear un botón para descargar el archivo
    st.download_button(
        label="Descargar archivo con predicciones",
        data=result_csv,
        file_name="archivo_con_predicciones.csv",
        mime="text/csv"
    )
