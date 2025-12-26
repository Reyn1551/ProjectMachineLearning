# Telco Customer Churn Prediction 📊

A Machine Learning application to predict customer churn in the telecommunications industry using Random Forest and K-Nearest Neighbors (KNN). Built with **Streamlit** for an interactive experience.

---

## 🇬🇧 English Documentation

### Overview
This project aims to identify customers who are likely to cancel their subscription (churn). By analyzing customer demographics, services, and account information, the application provides business insights and strategic recommendations to improve customer retention.

### Key Features
1.  **Data Exploration**: Interactive visualization of the dataset (Distribution charts, imbalance checks).
2.  **Preprocessing Pipeline**: Automated data cleaning, one-hot encoding, and scaling with visual "Before & After" differentiation.
3.  **Model Training**: Train **Random Forest** and **KNN** models directly from the UI, with optional **GridSearchCV** for hyperparameter tuning.
4.  **Results Comparison**: Side-by-side comparison of models using Accuracy, Precision, Recall, F1-Score, ROC Curves, and Confusion Matrices.
5.  **Business Insights**: A calculator to predict churn risk for new customers and generate actionable business recommendations (e.g., offering specific discounts).

### How to Run
1.  Ensure you have Python installed.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If using the provided `.venv`, dependencies are pre-installed).*
3.  Run the application:
    ```bash
    streamlit run app.py
    ```

---

## 🇮🇩 Dokumentasi Bahasa Indonesia

### Ringkasan
Proyek ini bertujuan untuk mengidentifikasi pelanggan yang berpotensi berhenti berlangganan (churn). Dengan menganalisis demografi, layanan yang dipakai, dan tagihan pelanggan, aplikasi ini memberikan wawasan bisnis dan rekomendasi strategi untuk mempertahankan pelanggan.

### Fitur Utama
1.  **Eksplorasi Data**: Visualisasi data interaktif (Grafik distribusi, pengecekan ketimpangan data).
2.  **Pipeline Preprocessing**: Pembersihan data otomatis, one-hot encoding, dan scaling dengan tampilan "Sebelum & Sesudah".
3.  **Pelatihan Model**: Melatih model **Random Forest** dan **KNN** langsung dari antarmuka, dilengkapi ospi **GridSearchCV** untuk optimasi otomatis.
4.  **Perbandingan Hasil**: Membandingkan performa model menggunakan Akurasi, Presisi, Recall, F1-Score, Kurva ROC, dan Confusion Matrix.
5.  **Wawasan Bisnis (Business Insights)**: Kalkulator untuk memprediksi risiko churn pelanggan baru dan memberikan saran bisnis yang konkret (misal: penawaran diskon khusus).

### Cara Menjalankan
1.  Pastikan Python sudah terinstal.
2.  Install library yang dibutuhkan:
    ```bash
    pip install -r requirements.txt
    ```
    *(Catatan: Jika menggunakan `.venv` yang tersedia, library sudah terinstall).*
3.  Jalankan aplikasi:
    ```bash
    streamlit run app.py
    ```

---
**Tech Stack**: Python, Streamlit, Scikit-Learn, Pandas, Plotly.
