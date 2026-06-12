import streamlit as st
import pickle
import numpy as np

# IMPORT FUNGSI DARI FILE PREPROCESSING
from preprocessing import download_nltk_resources, clean_text, word_tokenize, buang_stopword

#  KONFIGURASI HALAMAN & TAMPILAN
st.set_page_config(
    page_title="Analisis Sentimen Amazon Fine Food",
    page_icon="🍲",
    layout="centered"
)

st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    .stButton>button {
        width: 100%; background-color: #1E3A8A; color: white;
        border-radius: 8px; height: 3em; font-weight: bold;
    }
    .stButton>button:hover { background-color: #3B82F6; color: white; }
    .metric-box {
        padding: 15px; border-radius: 10px;
        background-color: #f3f4f6; border: 1px solid #e5e7eb;
    }
    </style>
""", unsafe_allow_html=True)

# Eksekusi pemuatan otomatis resource NLTK
download_nltk_resources()

#  MEMUAT ASSET MODEL & TF-IDF
@st.cache_resource
def load_machine_learning_assets():
    """Memuat model Logistic Regression dan objek TF-IDF Vectorizer"""
    with open('logistic_regression_model.pickle', 'rb') as f:
        model = pickle.load(f)
        
    with open('tfidf_vectorizer.pickle', 'rb') as f:
        tfidf = pickle.load(f)
        
    return model, tfidf

# Panggil fungsi load asset
model, tfidf = load_machine_learning_assets()

#  DESAIN ANTARMUKA / UI DESIGN
st.title("🍲 Amazon Fine Food Sentiment Analyzer")
st.caption("Aplikasi Kecerdasan Buatan Berbasis Logistic Regression & TF-IDF untuk Klasifikasi Opini Produk Makanan & Minuman.")
st.write("---")

st.markdown("##### **Masukkan Teks Ulasan Makanan / Minuman (Bahasa Inggris):**")
user_input = st.text_area(
    label="Input Teks",
    placeholder="Contoh: This coffee tastes amazing! It has a rich flavor and the packaging was perfect...",
    height=120,
    label_visibility="collapsed"
)

tombol_analisis = st.button("Analisis Sentimen Ulasan")

#  PIPELINE INFERENSI
if tombol_analisis:
    if user_input.strip() == "":
        st.warning("⚠️ Mohon ketikkan teks ulasan terlebih dahulu sebelum menekan tombol!")
    else:
        with st.spinner("Sedang memproses logika pembobotan kata (TF-IDF) & regresi..."):
            
            # 1. Case Folding & Cleaning
            teks_lower = user_input.lower()
            teks_clean = clean_text(teks_lower)
            
            # 2. Tokenisasi
            tokens = word_tokenize(teks_clean)
            
            # 3. Stopword Removal
            tokens_final = buang_stopword(tokens)
            teks_siap_tfidf = ' '.join(tokens_final)
            
            # 4. Transformasi Teks Menjadi Vektor Numerik Menggunakan TF-IDF
            vektor_tfidf = tfidf.transform([teks_siap_tfidf])
            
            # 5. Prediksi Probabilitas lewat Logistic Regression
            probabilitas = model.predict_proba(vektor_tfidf)[0][1]
            
            # 6. Tampilkan Hasil Output
            st.write("### **Hasil Analisis Model:**")
            
            if probabilitas > 0.5:
                tingkat_keyakinan = probabilitas * 100
                st.success(f"### SENTIMEN POSITIF (Pujian / Rekomendasi)")
                st.markdown(f"Model memprediksi ulasan ini berkonotasi **PUJIAN** dengan tingkat kepastian sebesar **{tingkat_keyakinan:.2f}%**.")
            else:
                tingkat_keyakinan = (1 - probabilitas) * 100
                st.error(f"### SENTIMEN NEGATIF (Keluhan / Kekecewaan)")
                st.markdown(f"Model memprediksi ulasan ini berkonotasi **KELUHAN** dengan tingkat kepastian sebesar **{tingkat_keyakinan:.2f}%**.")
            
            st.write("---")
            
            #  TRANSPARANSI PIPELINE LOG
            with st.expander("🔍 Lihat Detail Pemrosesan Internal Model (Pipeline Log)"):
                st.markdown("Berikut adalah transformasi data teks Anda dari data mentah hingga dihitung nilainya oleh model Logistic Regression:")
                st.info(f"**1. Teks Asli Pengguna:**\n*{user_input}*")
                st.info(f"**2. Hasil Text Cleaning & Tokenisasi:**\n*{tokens if tokens else '[Teks kosong setelah dibersihkan]'}*")
                st.info(f"**3. Hasil Akhir Setelah Stopword Removal:**\n*{teks_siap_tfidf if teks_siap_tfidf else '[Tidak ada kata tersisa]'}*")
                
                # Mendapatkan bobot kata yang aktif pada kalimat ini untuk transparansi nilai TF-IDF
                feature_names = tfidf.get_feature_names_out()
                non_zero_indices = vektor_tfidf.nonzero()[1]
                
                log_tfidf_kata = [f"{feature_names[i]}: {vektor_tfidf[0, i]:.4f}" for i in non_zero_indices]
                
                st.info(f"**4. Nilai Fitur TF-IDF yang Terdeteksi:**\n{', '.join(log_tfidf_kata) if log_tfidf_kata else 'Tidak ada kata yang cocok dengan kamus kosa kata makanan pada model.'}")