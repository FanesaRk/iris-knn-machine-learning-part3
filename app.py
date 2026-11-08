"""Antarmuka Streamlit: klasifikasi bunga Iris dari model .pth."""
from pathlib import Path

import pandas as pd
import streamlit as st

from knn_model import IrisKNN

st.set_page_config(page_title='Iris Classifier | KNN', page_icon='🌸', layout='centered')

MODEL = Path(__file__).resolve().parent / 'models' / 'iris_knn_tuned.pth'

@st.cache_resource
# Model hanya dibaca untuk inferensi; aman digunakan ulang di tiap rerun Streamlit.
def load_model():
    return IrisKNN(MODEL)

st.title('🌸 Klasifikasi Bunga Iris')
st.caption('Tugas Besar Machine Learning - Part 3 | Streamlit + KNN + PyTorch .pth')
st.write('Masukkan empat ukuran bunga dalam sentimeter (cm), lalu klik **Prediksi Jenis Iris**.')

if not MODEL.is_file():
    st.error('File `models/iris_knn_tuned.pth` belum tersedia. Jalankan notebook Colab terlebih dahulu.')
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f'Model gagal dimuat: {error}')
    st.stop()

with st.form('iris_form'):
    left, right = st.columns(2)
    with left:
        sepal_length = st.number_input('Panjang sepal (cm)', min_value=0.0, max_value=15.0, value=5.1, step=0.1)
        sepal_width = st.number_input('Lebar sepal (cm)', min_value=0.0, max_value=15.0, value=3.5, step=0.1)
    with right:
        petal_length = st.number_input('Panjang petal (cm)', min_value=0.0, max_value=15.0, value=1.4, step=0.1)
        petal_width = st.number_input('Lebar petal (cm)', min_value=0.0, max_value=15.0, value=0.2, step=0.1)
    submitted = st.form_submit_button('🔎 Prediksi Jenis Iris', use_container_width=True)

if submitted:
    try:
        result, votes = model.predict_one([sepal_length, sepal_width, petal_length, petal_width])
    except ValueError as error:
        st.error(str(error))
    else:
        st.success(f'Hasil klasifikasi: **Iris {result.capitalize()}**')
        st.subheader('Proporsi suara tetangga KNN')
        chart = pd.DataFrame({'Jenis Bunga': [f'Iris {x.capitalize()}' for x in votes],
                              'Proporsi Suara': list(votes.values())}).set_index('Jenis Bunga')
        st.bar_chart(chart, y='Proporsi Suara')
        st.info('Proporsi suara menunjukkan jumlah tetangga terdekat per kelas; ini bukan probabilitas yang telah dikalibrasi.')

with st.expander('Tentang model'):
    st.write(f'Model: K-Nearest Neighbors | k = {model.k} | p = {model.p} (Euclidean jika p=2)')
    st.write(f'Jumlah data training tersimpan: {len(model.y)} sampel.')
    st.write('Model dan parameter scaler berada di berkas `models/iris_knn_tuned.pth`.')
    st.write('Sumber data: dataset Iris pada Scikit-learn. Penggunaan: demo akademik.')
