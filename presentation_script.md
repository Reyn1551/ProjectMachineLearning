# Naskah Presentasi: Telco Customer Churn Prediction (Versi Streamlit)

Dokumen ini berisi panduan langkah demi langkah untuk mempresentasikan aplikasi Streamlit Anda. Terdapat 3 bagian untuk setiap halaman:
1.  **Aksi**: Apa yang harus Anda klik/tunjukkan di layar.
2.  **Narasi Presentasi**: Apa yang Anda ucapkan kepada audiens.
3.  **Detail Teknis**: Penjelasan mendalam jika ada pertanyaan "mengapa/bagaimana".

---

## 1. Pembukaan

**Narasi:**
"Halo semuanya. Hari ini saya akan mempresentasikan proyek Machine Learning saya tentang **Prediksi Customer Churn** di industri Telekomunikasi. Tujuan proyek ini adalah membangun model yang bisa membedakan mana pelanggan yang setia dan mana yang berpotensi pindah (churn), serta memberikan rekomendasi bisnis untuk menahannya. Saya telah membangun aplikasi web interaktif untuk mendemonstrasikan proses end-to-end dari data mentah hingga prediksi bisnis."

---

## 2. Halaman 1: Data Exploration

**Aksi:**
*   Buka aplikasi Streamlit.
*   Pastikan ada di halaman **"1. Data Exploration"**.
*   Klik tombol **"Load Default Dataset"** jika data belum muncul.
*   Scroll ke bawah perlahan sambil menjelaskan grafik.

**Narasi:**
"Langkah pertama adalah memahami data kita. Dataset ini berisi profil pelanggan Telco.
*   **Dataset Overview**: Kita memiliki 7043 baris data pelanggan dengan 21 atribut.
*   **Target Variable**: Grafik Pie Chart ini menunjukkan kolom 'Churn'. Kita bisa lihat dataset ini **imbalance** (tidak seimbang), di mana mayoritas pelanggan adalah 'No' (tidak churn), dan hanya sebagian kecil yang 'Yes'.
*   **Distribusi Fitur**: Di sebelah kanan, kita melihat 'Payment Method'. Terlihat bahwa pelanggan yang menggunakan *Electronic Check* memiliki tingkat churn yang sangat tinggi dibandingkan metode lain."

**Detail Teknis:**
*   *Mengapa Imbalance itu penting?* Karena jika data target tidak seimbang (misal 73% No : 27% Yes), model cenderung bias ke kelas mayoritas. Akurasi 73% bisa saja menipu (karena model hanya menebak "No" untuk semua orang). Nanti kita akan melihat metrik lain seperti F1-Score atau Recall yang lebih jujur.

---

## 3. Halaman 2: Data Preprocessing

**Aksi:**
*   Pindah ke menu **"2. Preprocessing"** di sidebar.
*   Jelaskan langkah-langkah di teks kiri.
*   Klik tombol **"🚀 Run Preprocessing Pipeline"**.
*   Tunjukkan perbedaan tabel "Before" dan "After".

**Narasi:**
"Data mentah tidak bisa langsung masuk ke model ML. Kita perlu membersihkannya melalui **Preprocessing Pipeline**:
1.  **Cleaning**: Kolom `TotalCharges` awalnya string / teks karena ada spasi kosong. Kita ubah jadi angka dan isi nilai kosongnya dengan rata-rata.
2.  **Encoding**: Mesin hanya mengerti angka, bukan teks. Jadi kolom kategori seperti `Gender` atau `Partner` kita ubah jadi angka menggunakan teknik *One-Hot Encoding*.
3.  **Scaling**: Rentang nilai `Tenure` (0-72 bulan) dan `TotalCharges` (ribuan dollar) sangat berbeda. Kita standarisasi menggunakan *StandardScaler* agar semua fitur punya skala yang sama.
4.  **Splitting**: Kita bagi data jadi 80% untuk Training dan 20% untuk Testing."

**Detail Teknis:**
*   *One-Hot Encoding*: Mengubah 1 kolom kategori menjadi banyak kolom biner (0/1). Contoh: `Color: Red` menjadi `Color_Red: 1, Color_Blue: 0`.
*   *StandardScaler*: Mengubah data sehingga rata-ratanya 0 dan standar deviasinya 1. Penting untuk algoritma berbasis jarak seperti KNN.

---

## 4. Halaman 3: Model Training

**Aksi:**
*   Pindah ke menu **"3. Model Training"**.
*   Centang checkbox **"Enable GridSearchCV"** (opsional, katakan ini untuk performa terbaik).
*   Klik **"Start Training"**.
*   Tunggu progress bar selesai.

**Narasi:**
"Di sini kita melatih dua algoritma populer sebagai perbandingan:
1.  **Random Forest**: Algoritma berbasis *Ensemble* yang menggabungkan banyak pohon keputusan. Sangat kuat dan tahan terhadap overfitting.
2.  **K-Nearest Neighbors (KNN)**: Algoritma sederhana yang mengklasifikasikan data berdasarkan 'tetangga' terdekatnya.

Fitur **GridSearchCV** yang saya aktifkan ini akan mencari kombinasi parameter terbaik secara otomatis, misalnya mencari jumlah pohon optimal untuk Random Forest."

**Detail Teknis:**
*   *GridSearchCV*: Melakukan pencarian *brute-force* pada grid parameter yang kita tentukan untuk menemukan settingan model yang memberikan akurasi tertinggi (Cross-Validation).

---

## 5. Halaman 4: Results & Comparison

**Aksi:**
*   Pindah ke menu **"4. Results & Comparison"**.
*   Tunjukkan tabel metrics.
*   Buka Tab **"ROC Curves"** dan **"Confusion Matrix"**.
*   Terakhir, buka Tab **"Feature Importance"** (Highlight bagian ini!).

**Narasi:**
"Setelah training selesai, mari kita lihat hasilnya.
*   **Metrics**: Secara umum, Random Forest (batang biru/oranye) menang dalam Akurasi dan Presisi dibandingkan KNN.
*   **ROC Curve**: Kurva Random Forest lebih mendekati pojok kiri atas (AUC area lebih besar), yang berarti model ini lebih pintar membedakan churn vs non-churn.
*   **Feature Importance**: Ini adalah insight paling berharga. Kita bisa lihat bahwa **'Monthly Charges', 'Total Charges', dan 'Tenure'** adalah faktor paling krusial yang menentukan apakah orang akan pindah atau tidak."

**Detail Teknis:**
*   *Precision*: Dari semua yang *diprediksi* churn, berapa yang benar? (Penting agar tidak spam promo ke orang salah).
*   *Recall*: Dari semua yang *sebenarnya* churn, berapa yang berhasil dideteksi? (Penting agar tidak kecolongan pelanggan kabur).
*   *F1-Score*: Rata-rata harmonik Precision dan Recall.

---

## 6. Halaman 5: Business Insight (Demo)

**Aksi:**
*   Pindah ke menu **"5. Business Insights"**.
*   Isi form dengan data simulasi pelanggan berisiko tinggi:
    *   *Tenure*: **2** (baru berlangganan)
    *   *Monthly*: **100** (mahal)
    *   *Total*: **200**
    *   *Contract*: **Month-to-month**
    *   *Payment*: **Electronic check**
*   Klik **"Predict Churn Risk"**.
*   Baca rekomendasi yang muncul.

**Narasi:**
"Bagian terakhir adalah implementasi bisnis. Bayangkan kita adalah Customer Service.
Jika ada pelanggan baru dengan tagihan mahal dan kontrak bulanan seperti yang saya input ini, sistem akan memprediksi **High Churn Risk**.
Sistem langsung memberikan rekomendasi strategis:
1.  Tawarkan diskon untuk pindah ke kontrak 1 tahun.
2.  Arahkan pembayaran ke Auto-pay untuk mengurangi friksi transaksi.
Ini membuktikan model kita tidak hanya akurat secara matematis, tapi juga **actionable** secara bisnis."

---

## 7. Penutup

**Narasi:**
"Kesimpulannya, proyek ini berhasil membangun sistem prediksi churn yang andal menggunakan Random Forest. Dengan aplikasi ini, perusahaan Telco bisa mendeteksi pelanggan berisiko lebih dini dan mengambil tindakan pencegahan yang tepat sasaran. Terima kasih."
