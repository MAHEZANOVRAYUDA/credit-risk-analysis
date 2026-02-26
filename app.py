import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Konfigurasi Halaman
st.set_page_config(
    page_title="Maheza Novrayuda Analyzer",
    page_icon="🔍",
    layout="centered",
)

# Kustomisasi CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .stApp {
        background-color: #1a202c;
        color: #ffffff;
    }

    label {
        color: #ffffff !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        margin-bottom: 5px !important;
    }

    hr {
        border: 1px solid #4a5568;
        margin: 2rem 0;
    }

    .result-card {
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
    }
    
    .good-risk {
        background-color: #2f855a;
        color: #ffffff;
        border: 2px solid #48bb78;
    }
    
    .bad-risk {
        background-color: #9b2c2c;
        color: #ffffff;
        border: 2px solid #f56565;
    }

    .stButton>button {
        background-color: #3182ce !important;
        color: white !important;
        font-weight: bold !important;
        height: 3em !important;
        width: 100% !important;
        border-radius: 8px !important;
        font-size: 1.1rem !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Memuat model machine learning dari file joblib."""
    try:
        return joblib.load('credit_risk_model_pack.joblib')
    except Exception:
        return None

# Inisialisasi Model
model_pack = load_model()

# Header Aplikasi
st.title("Credit risk Analysis")
st.write("Sistem prediksi risiko kredit nasabah berdasarkan data historis.")
st.markdown("---")

if model_pack is None:
    st.error("File 'credit_risk_model_pack.joblib' tidak ditemukan.")
else:
    # Form Input: Profil Keuangan
    st.header("1. Profil Keuangan")
    
    col1, col2 = st.columns(2)
    with col1:
        grade = st.selectbox("Grade Kualitas Pinjaman", ["A", "B", "C", "D", "E", "F", "G"])
        loan_amnt = st.number_input("Jumlah Pinjaman ($)", min_value=500, value=15000, step=500)
        int_rate = st.number_input("Suku Bunga (%)", min_value=5.0, max_value=40.0, value=12.0, step=0.1)

    with col2:
        annual_inc = st.number_input("Pendapatan Tahunan ($)", min_value=0, value=50000, step=1000)
        dti = st.number_input("Rasio Hutang (DTI %)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
        verification = st.selectbox("Status Verifikasi", ['Verified', 'Source Verified', 'Not Verified'])

    st.markdown("---")
    
    # Form Input: Riwayat Monitoring
    st.header("2. Riwayat Monitoring")
    
    col3, col4 = st.columns(2)
    with col3:
        total_pymnt = st.number_input("Total Terbayar ($)", min_value=0.0, value=10000.0)
        out_prncp = st.number_input("Sisa Pokok ($)", min_value=0.0, value=5000.0)
        
    with col4:
        last_pymnt = st.number_input("Pembayaran Terakhir ($)", min_value=0.0, value=1000.0)
        emp_len = st.selectbox("Lama Bekerja", [
            '10+ years', '9 years', '8 years', '7 years', '6 years', 
            '5 years', '4 years', '3 years', '2 years', '1 year', '< 1 year'
        ])

    st.markdown("---")

    # Logika Prediksi
    if st.button("JALANKAN ANALISIS"):
        # Penyiapan fitur sesuai format model
        input_features = {f: 0 for f in model_pack['features']}
        
        # Mapping input user ke fitur model
        input_features['grade'] = model_pack['ordinal_maps']['grade'].get(grade, 3)
        input_features['loan_amnt'] = loan_amnt
        input_features['int_rate'] = int_rate
        input_features['annual_inc'] = annual_inc
        input_features['dti'] = dti
        input_features['verification_status'] = model_pack['ordinal_maps']['verification_status'].get(verification, 0)
        input_features['total_pymnt'] = total_pymnt
        input_features['out_prncp'] = out_prncp
        input_features['last_pymnt_amnt'] = last_pymnt
        input_features['emp_length'] = model_pack['ordinal_maps']['emp_length'].get(emp_len, 0)
        
        # Penanganan data leakage (recoveries)
        if 'recoveries' in input_features:
            input_features['recoveries'] = 0

        # Eksekusi Prediksi
        df_input = pd.DataFrame([input_features])[model_pack['features']]
        scaled_input = model_pack['scaler'].transform(df_input)
        
        prob_good = model_pack['model'].predict_proba(scaled_input)[0][1]
        is_good = model_pack['model'].predict(scaled_input)[0]

        # Tampilan Hasil
        st.subheader("Hasil Analisis:")
        
        if is_good == 1:
            st.markdown(f"""
                <div class="result-card good-risk">
                    <h2 style='margin:0;'>PROFIL: GOOD RISK</h2>
                    <p style='font-size:1.2rem; margin-top:10px;'>Nasabah layak diberikan kredit.</p>
                    <p>Tingkat Keyakinan: {prob_good*100:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
            st.success("Analisis selesai. Parameter keuangan menunjukkan profil risiko yang rendah.")
        else:
            st.markdown(f"""
                <div class="result-card bad-risk">
                    <h2 style='margin:0;'>PROFIL: BAD RISK</h2>
                    <p style='font-size:1.2rem; margin-top:10px;'>Peringatan: Risiko gagal bayar tinggi.</p>
                    <p>Skor Keamanan: {prob_good*100:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
            st.warning("Perhatian: Rasio keuangan berada dalam kategori risiko tinggi.")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("Aplikasi Prediksi - @Maheza Novrayuda")