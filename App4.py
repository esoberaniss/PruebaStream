import streamlit as st

# Se inicializa la variable como False en cada ejecución
verified = False  

if st.button("Verify"):
    verified = True  # Si se presiona el botón, cambiamos a True

if verified:
    st.write("Verification successful!")
    st.selectbox("Phase",options=("Hola","Mundo"))
