# Credit Risk Analysis & Prediction

Proyek ini bertujuan untuk menganalisis risiko kredit menggunakan dataset historis (2007-2014) dan membangun model prediksi yang diintegrasikan dengan aplikasi web berbasis **Streamlit**.

## Fitur Utama
- **EDA & Analisis Data**: Pembersihan data, visualisasi korelasi, dan analisis fitur penting.
- **Model Machine Learning**: Menggunakan algoritma **Gradient Boosting** (dan perbandingan dengan XGBoost, Random Forest, dll).
- **Aplikasi Web (Streamlit)**: Antarmuka modern untuk melakukan prediksi risiko kredit nasabah secara instan.

## Struktur Project
- `Final_Task_IDX_Partners_Data_Scientist_CreditRisk.ipynb`: Notebook untuk analisis dan pelatihan model.
- `app.py`: Kode utama aplikasi Streamlit.
- `credit_risk_model_pack.joblib`: Model yang sudah dilatih dan siap digunakan.
- `requirements.txt`: Daftar library yang dibutuhkan.

## Cara Menjalankan Secara Lokal
1. Clone repository ini.
2. Install depedensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```

## Dataset
*Dataset asli (`loan_data_2007_2014.csv`) tidak disertakan karena ukuran yang besar.*
