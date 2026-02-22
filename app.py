import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Credit Risk Analyzer",
    page_icon="�",
    layout="centered",
)

# 2. STYLE: FOKUS PADA READABILITY (Sangat Mudah Dibaca)
st.markdown("""
    <style>
    /* Menggunakan font standar yang bersih */
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Latar belakang gelap yang solid untuk kontras teks putih */
    .stApp {
        background-color: #1a202c;
        color: #ffffff;
    }

    /* Label Input: Besar dan Terang */
    label {
        color: #ffffff !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        margin-bottom: 5px !important;
    }

    /* Garis pemisah */
    hr {
        border: 1px solid #4a5568;
        margin: 2rem 0;
    }

    /* Box Hasil */
    .result-card {
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
    }
    .good-risk {
        background-color: #2f855a; /* Hijau Tua */
        color: #ffffff;
        border: 2px solid #48bb78;
    }
    .bad-risk {
        background-color: #9b2c2c; /* Merah Tua */
        color: #ffffff;
        border: 2px solid #f56565;
    }

    /* Button */
    .stButton>button {
        background-color: #3182ce !important; /* Biru Solid */
        color: white !important;
        font-weight: bold !important;
        height: 3em !important;
        width: 100% !important;
        border-radius: 8px !important;
        font-size: 1.1rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. FUNGSI LOAD MODEL
@st.cache_resource
def load_model_pack():
    try:
        return joblib.load('credit_risk_model_pack.joblib')
    except Exception as e:
        return None

pack = load_model_pack()

# 4. TAMPILAN UTAMA
st.title("Analisis Risiko Kredit")
st.write("Masukkan data di bawah ini untuk memprediksi tingkat kelayakan kredit.")
st.markdown("---")

if pack is None:
    st.error("File model 'credit_risk_model_pack.joblib' tidak ditemukan di folder ini.")
else:
    # --- SECTION 1: PROFIL NASABAH (Data Utama) ---
    st.header("1. Profil Keuangan & Aplikasi")
    
    col1, col2 = st.columns(2)
    with col1:
        grade = st.selectbox("Grade (Kualitas Pinjaman)", ["A", "B", "C", "D", "E", "F", "G"])
        loan_amnt = st.number_input("Jumlah Pinjaman ($)", min_value=500, value=15000, step=500)
        int_rate = st.number_input("Suku Bunga (%)", min_value=5.0, max_value=40.0, value=12.0, step=0.1)

    with col2:
        annual_inc = st.number_input("Pendapatan Tahunan ($)", min_value=0, value=50000, step=1000)
        dti = st.number_input("Rasio Hutang (DTI %)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
        verify_stat = st.selectbox("Status Verifikasi", ['Verified', 'Source Verified', 'Not Verified'])

    st.markdown("---")
    
    # --- SECTION 2: RIWAYAT PEMBAYARAN (Gunakan 0 jika nasabah baru) ---
    st.header("2. Riwayat & Transaksi (Monitoring)")
    st.info("Jika nasabah baru, biarkan nilai default atau nol.")
    
    col3, col4 = st.columns(2)
    with col3:
        total_pymnt = st.number_input("Total yang Sudah Terbayar ($)", min_value=0.0, value=10000.0)
        out_prncp = st.number_input("Sisa Pinjaman Pokok ($)", min_value=0.0, value=5000.0)
        
    with col4:
        last_pymnt_amnt = st.number_input("Jumlah Pembayaran Terakhir ($)", min_value=0.0, value=1000.0)
        emp_length = st.selectbox("Lama Bekerja", [
            '10+ years', '9 years', '8 years', '7 years', '6 years', 
            '5 years', '4 years', '3 years', '2 years', '1 year', '< 1 year'
        ])

    st.markdown("---")

    # 5. TOMBOL PREDIKSI
    if st.button("MULAI ANALISIS SEKARANG"):
        # Menyiapkan data untuk model
        input_data = {feat: 0 for feat in pack['features']}
        
        # Mapping input ke fitur yang sesuai
        input_data['grade'] = pack['ordinal_maps']['grade'].get(grade, 3)
        input_data['loan_amnt'] = loan_amnt
        input_data['int_rate'] = int_rate
        input_data['annual_inc'] = annual_inc
        input_data['dti'] = dti
        input_data['verification_status'] = pack['ordinal_maps']['verification_status'].get(verify_stat, 0)
        input_data['total_pymnt'] = total_pymnt
        input_data['out_prncp'] = out_prncp
        input_data['last_pymnt_amnt'] = last_pymnt_amnt
        input_data['emp_length'] = pack['ordinal_maps']['emp_length'].get(emp_length, 0)
        
        # Fitur recoveries diisi 0 karena fitur ini data leakage jika diisi manual
        if 'recoveries' in input_data:
            input_data['recoveries'] = 0

        # Memproses Prediksi
        df_input = pd.DataFrame([input_data])[pack['features']]
        scaled_input = pack['scaler'].transform(df_input)
        
        proba = pack['model'].predict_proba(scaled_input)[0][1]
        prediction = pack['model'].predict(scaled_input)[0]

        # 6. TAMPILAN HASIL
        st.subheader("Hasil Analisis:")
        
        if prediction == 1:
            st.markdown(f"""
                <div class="result-card good-risk">
                    <h2 style='margin:0;'>PROFIL: GOOD RISK</h2>
                    <p style='font-size:1.2rem; margin-top:10px;'>Nasabah Layak Diberikan Pinjaman</p>
                    <p>Tingkat Keyakinan: {proba*100:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
            st.success("Analisis selesai. Parameter keuangan nasabah menunjukkan perilaku pembayaran yang stabil.")
        else:
            st.markdown(f"""
                <div class="result-card bad-risk">
                    <h2 style='margin:0;'>PROFIL: BAD RISK</h2>
                    <p style='font-size:1.2rem; margin-top:10px;'>Peringatan: Risiko Gagal Bayar Tinggi</p>
                    <p>Skor Keamanan: {proba*100:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
            st.warning("Perhatian: Rasio keuangan nasabah dikategorikan berisiko tinggi.")

st.markdown("<br><br><br>", unsafe_allow_html=True)
st.caption("Aplikasi Prediksi Risiko Kredit v3.0 | Sederhana & Kontras Tinggi")