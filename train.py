
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
import joblib
import os

# --- Konfigurasi ---
DATA_FILE = 'Telco-Customer-Churn.csv'
MODEL_FILE = 'model.joblib'
SCALER_FILE = 'scaler.joblib'
COLUMNS_FILE = 'columns.joblib'

def train_model():
    """
    Fungsi ini memuat data, melakukan pra-pemrosesan, melatih model Random Forest,
    dan menyimpan artefak yang diperlukan (model, scaler, dan daftar kolom).
    """
    print(f"Mencoba memuat data dari {DATA_FILE}...")
    try:
        df = pd.read_csv(DATA_FILE)
        print("Dataset berhasil dimuat.")
    except FileNotFoundError:
        print("---------------------------------------------------------------")
        print(f"PERINGATAN: File data '{DATA_FILE}' tidak ditemukan.")
        print("Membuat model dummy dan artefak placeholder.")
        print("Aplikasi GUI akan berjalan, tetapi prediksinya tidak akan akurat.")
        print(f"Silakan letakkan '{DATA_FILE}' di direktori yang sama dengan skrip ini dan jalankan kembali 'python train.py' untuk melatih model yang sebenarnya.")
        print("---------------------------------------------------------------")
        create_dummy_artifacts()
        return

    # --- Pra-pemrosesan Data (diadaptasi dari skrip asli) ---
    print("Memulai pra-pemrosesan data...")
    # Mengatasi nilai non-numerik di 'TotalCharges'
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

    # Menentukan kolom kategorikal dan numerik
    categorical_cols = [col for col in df.select_dtypes(include='object').columns if col not in ['customerID', 'Churn']]
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # 'SeniorCitizen' adalah numerik tetapi sebenarnya kategorikal
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)
    categorical_cols.append('SeniorCitizen')

    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Menghapus 'customerID' yang tidak relevan
    if 'customerID' in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=['customerID'])

    # Memisahkan fitur (X) dan target (y)
    X = df_encoded.drop(columns=['Churn'])
    y = df_encoded['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Menyelaraskan kolom (menghapus kolom target jika ada secara tidak sengaja)
    if 'Churn_Yes' in X.columns:
         X = X.drop(columns=['Churn_Yes'])

    # --- Penskalaan Fitur Numerik ---
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    print("Pra-pemrosesan data selesai.")

    # --- Pelatihan Model ---
    print("Memulai pelatihan model Random Forest...")
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    print("Pelatihan model selesai.")

    # --- Menyimpan Artefak ---
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(X.columns.tolist(), COLUMNS_FILE)
    print(f"Model telah disimpan ke {MODEL_FILE}")
    print(f"Scaler telah disimpan ke {SCALER_FILE}")
    print(f"Kolom telah disimpan ke {COLUMNS_FILE}")

def create_dummy_artifacts():
    """
    Fungsi ini membuat dan menyimpan artefak placeholder jika dataset asli tidak ditemukan.
    Ini memungkinkan aplikasi Streamlit untuk berjalan tanpa kesalahan.
    """
    # Membuat DataFrame tiruan yang strukturnya mirip dengan data asli
    dummy_data = {
        'tenure': [10, 20], 'MonthlyCharges': [50, 70], 'TotalCharges': [500, 1400],
        'gender_Male': [0, 1], 'Partner_Yes': [1, 0], 'Dependents_Yes': [0, 0],
        'PhoneService_Yes': [1, 1], 'MultipleLines_No': [1, 0], 'MultipleLines_Yes': [0, 1],
        'InternetService_DSL': [1, 0], 'InternetService_Fiber optic': [0, 1], 'OnlineSecurity_Yes': [1, 0],
        'OnlineBackup_Yes': [0, 1], 'DeviceProtection_No': [1, 0], 'TechSupport_Yes': [0, 1],
        'StreamingTV_No': [1, 0], 'StreamingMovies_No': [0, 1], 'Contract_Month-to-month': [1, 0],
        'Contract_One year': [0, 1], 'PaperlessBilling_Yes': [1, 1], 'PaymentMethod_Electronic check': [1, 0],
        'SeniorCitizen_0': [1, 0], 'SeniorCitizen_1': [0, 1]
    }
    # Tambahkan beberapa kolom lagi untuk mencocokkan kemungkinan output one-hot encoding
    all_possible_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check', 'SeniorCitizen_1']
    dummy_df = pd.DataFrame(dummy_data)
    
    # Buat kolom yang hilang dan isi dengan 0
    final_cols = []
    for col in all_possible_cols:
        if col in dummy_df.columns:
            final_cols.append(col)
        # Menambahkan kolom tiruan jika tidak ada di dummy_data
        # Ini untuk memastikan daftar kolom yang disimpan cukup komprehensif
    
    # Simpan daftar kolom tiruan
    dummy_columns = final_cols + [c for c in all_possible_cols if c not in final_cols]

    # Buat dan simpan scaler tiruan
    scaler = StandardScaler()
    dummy_numerical = dummy_df[['tenure', 'MonthlyCharges', 'TotalCharges']]
    scaler.fit(dummy_numerical)
    
    # Buat dan simpan model klasifikasi tiruan
    dummy_model = DummyClassifier(strategy="most_frequent")
    dummy_model.fit(dummy_numerical, [0, 1])

    # Simpan artefak tiruan
    joblib.dump(dummy_model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(dummy_columns, COLUMNS_FILE)
    print("Artefak DUMMY (model, scaler, kolom) berhasil dibuat.")

if __name__ == "__main__":
    train_model()
