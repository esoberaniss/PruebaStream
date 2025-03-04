import streamlit as st

st.write(st.session_state)

# Inicializar en session_state si no existe
if "verified" not in st.session_state:
    st.session_state.verified = False  

if st.button("Verify"):
    st.session_state.verified = True  # Guardar el estado

    if st.session_state.verified:
        st.write("Verification successful!")
        st.selectbox("Phase", options=("Hola", "Mundo"))

