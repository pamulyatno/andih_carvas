import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# --- 1. Memuat Model dan Objek Pra-pemrosesan ---

@st.cache_resource
def load_resources():
    """Memuat model dan preprocessor sekali saat aplikasi dimulai."""
    try:
        model = load_model('my_cardio_model.h5')
        preprocessor = joblib.load('my_data_preprocessor.pkl')
        label_encoder = joblib.load('my_label_encoder.pkl')
        return model, preprocessor, label_encoder
    except FileNotFoundError as e:
        st.error(f"Error: File resource tidak ditemukan. Pastikan 'my_cardio_model.h5', 'my_data_preprocessor.pkl', dan 'my_label_encoder.pkl' ada di folder yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"Error saat memuat resource: {e}")
        st.stop()

model, preprocessor, label_encoder = load_resources()

# --- 2. Judul Aplikasi ---
st.subheader("Aplikasi Prediksi Risiko Kardiovaskular")
st.markdown("Masukkan data pasien di bawah ini untuk memprediksi risiko penyakit kardiovaskular.")

# --- 3. Formulir Input Data ---
with st.form("input_form"):
    b1kol1,b1kol2,b1kol3,b1kol4 = st.columns(4) 
    # st.subheader("Data Pasien")

    # Kolom Numerik
    with b1kol1:
        st.markdown("Usia (tahun)      :")
        st.markdown("   ")
        st.markdown("Jenis Kelamin     :")
        st.markdown("   ")
        st.markdown("TD.Sistolik(hi)   :")
        st.markdown("   ")
        st.markdown("Kebiasaan Merokok :")
        st.markdown("   ")
        st.markdown("Aktivitas Fisik   :")
        st.markdown("   ")
        st.markdown("Tingkat Kolesterol:")
    with b1kol2:
        age_year = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=55,label_visibility="collapsed")
        gender = st.selectbox("Jenis Kelamin", options=["Pria", "Wanita"], index=0,label_visibility="collapsed")
        ap_hi = st.number_input("Sistolik (ap_hi)", min_value=1, value=120,label_visibility="collapsed")
        smoke = st.selectbox("Kebiasaan Merokok", options=["Tidak", "Ya"], index=0,label_visibility="collapsed")
        active = st.selectbox("Aktivitas Fisik", options=["Tidak", "Ya"], index=1,label_visibility="collapsed")
        cholesterol = st.selectbox("Tingkat Kolesterol", options=["Normal", "Di Atas Normal", "Sangat di Atas Normal"], index=0,label_visibility="collapsed")
    with b1kol3:
        st.markdown("Tinggi Badan(cm)   :")
        st.markdown("  ")
        st.markdown("Berat Badan (kg)   :")
        st.markdown("   ")
        st.markdown("TD.Diastolik(lo)   :")
        st.markdown("   ")
        st.markdown("Mengonsumsi Alkohol:")
        st.markdown("   ")
        st.markdown("Tingkat Glukosa  :")
    with b1kol4:
        height = st.number_input("TB", min_value=1, max_value=300, value=165,label_visibility="collapsed")
        weight = st.number_input("BB", min_value=1, max_value=300, value=70,label_visibility="collapsed")
        ap_lo = st.number_input("Diastolik", min_value=1, value=80,label_visibility="collapsed")
        alco = st.selectbox("Kebiasaan Mengonsumsi Alkohol", options=["Tidak", "Ya"], index=0,label_visibility="collapsed")
        gluc = st.selectbox("Tingkat Glukosa", options=["Normal", "Di Atas Normal", "Sangat di Atas Normal"], index=0,label_visibility="collapsed")
        submitted = st.form_submit_button("Prediksi")

    # Kolom Kategorikal
    # active = st.selectbox("Aktivitas Fisik", options=["Tidak", "Ya"], index=1)

    # Tombol Submit

# --- 4. Logika Prediksi ---
if submitted:
    # --- Pra-pemrosesan Data Baru ---

    # Mengubah input menjadi format yang dibutuhkan oleh model
    gender_map = {"Pria": 2, "Wanita": 1} # Sesuaikan dengan encoding di data training Anda
    cholesterol_map = {"Normal": 1, "Di Atas Normal": 2, "Sangat di Atas Normal": 3}
    gluc_map = {"Normal": 1, "Di Atas Normal": 2, "Sangat di Atas Normal": 3}
    binary_map = {"Tidak": 0, "Ya": 1}

    # Membuat DataFrame untuk data baru dengan fitur yang lengkap
    new_data = pd.DataFrame([{
        'height': height,
        'weight': weight,
        'ap_hi': ap_hi,
        'ap_lo': ap_lo,
        'age_year': age_year,
        'gender': gender_map[gender],
        'cholesterol': cholesterol_map[cholesterol],
        'gluc': gluc_map[gluc],
        'smoke': binary_map[smoke],
        'alco': binary_map[alco],
        'active': binary_map[active],
    }])

    # Menggunakan preprocessor yang telah dilatih untuk mentransformasi data baru
    new_data_processed = preprocessor.transform(new_data)

    # --- Prediksi ---
    prediction_proba = model.predict(new_data_processed)[0][0]
    prediction_class_encoded = (prediction_proba > 0.5).astype(int)
    prediction_label = label_encoder.inverse_transform([prediction_class_encoded])[0]

    # --- 5. Menampilkan Hasil ---
    st.subheader("Hasil Prediksi")

    if prediction_label == 1:
        result_text = "Tinggi"
        st.error(f"Hasil Prediksi: Risiko **{result_text}**")
        st.write("Pasien ini diprediksi memiliki risiko tinggi terkena penyakit kardiovaskular.")
    else:
        result_text = "Rendah"
        st.success(f"Hasil Prediksi: Risiko **{result_text}**")
        st.write("Pasien ini diprediksi memiliki risiko rendah terkena penyakit kardiovaskular.")

    st.info(f"Probabilitas risiko: {prediction_proba * 100:.2f}%")