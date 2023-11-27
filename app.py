import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict


st.title('Clasificando flores de Iris')
st.markdown('Modelo de juguete para clasificar flores de iris en \
     (setosa, versicolor, virginica) basándose en su sépalo/pétalo  \
    y longitud/ancho.')

st.header("Carasterísticas de las Plantas")
col1, col2 = st.columns(2)

with col1:
    st.text("Características del Sépalo")
    sepal_l = st.slider('Largo del Sépalo (cm)', 1.0, 8.0, 0.5)
    sepal_w = st.slider('Ancho del Sépalo (cm)', 2.0, 4.4, 0.5)

with col2:
    st.text("Características del Pétalo")
    petal_l = st.slider('Largo del Pépalo (cm)', 1.0, 7.0, 0.5)
    petal_w = st.slider('Ancho del Pépalo (cm)', 0.1, 2.5, 0.5)

st.text('')
if st.button("Predecir el tipo de Iris"):
    result = predict(
        np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
    st.text(result[0])


st.text('')
st.text('')

