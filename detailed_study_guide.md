# Panduan Pendalaman Materi & Pertanyaan Killer
**Dokumen Persiapan Sidang/Presentasi Machine Learning**

Dokumen ini disusun khusus untuk menghadapi pertanyaan mendalam dari dosen penguji "Killer". Pelajari setiap poinnya dan pahami *mengapa* (why) dan *bagaimana* (how), bukan hanya *apa* (what).

---

## 1. Bedah Dataset: Telco Customer Churn (Sangat Detail)

**Sumber Data**: Dataset ini adalah dataset standar industri (mirip IBM Telco Data) yang merekam perilaku 7.043 pelanggan telekomunikasi di California pada Q3.

**Struktur Data (7043 baris, 21 kolom):**
Dataset ini terdiri dari 3 blok informasi utama. Jika ditanya "Variabel apa saja yang mempengaruhi?", jawab dengan struktur ini:

### A. Demografi Pelanggan (Demographics)
Profil statis pelanggan. Biasanya pengaruhnya kecil terhadap Churn, tapi penting untuk segmentasi.
1.  `gender`: Pria/Wanita (Biasanya tidak berkorelasi kuat dengan churn).
2.  `SeniorCitizen`: Apakah lansia (0/1). *Insight*: Lansia seringkali lebih setia, ATAU justru lebih sulit beradaptasi dengan teknologi baru.
3.  `Partner`: Punya pasangan?
4.  `Dependents`: Punya tanggungan (anak/ortu)?
    *   *Logika*: Orang yang punya Partner/Dependents biasanya lebih **setia (Churn rendah)** karena enggan repot ganti provider untuk satu keluarga.

### B. Layanan yang Dipakai (Services)
Apa yang mereka beli. Ini menunjukkan seberapa "terikat" pelanggan.
5.  `PhoneService` & `MultipleLines`: Layanan telepon.
6.  `InternetService`: **DSL**, **Fiber Optic**, atau **No**.
    *   *Critical Insight*: Pengguna **Fiber Optic** seringkali memiliki Churn Rate TINGGI. Kenapa? Karena biasanya lebih mahal, atau kompetisi di *high-speed internet* lebih ketat, atau teknisnya sering bermasalah.
7.  `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`: Layanan tambahan (Add-ons).
    *   *Logika*: Semakin banyak add-ons yang diambil, semakin **sulit** pelanggan untuk pindah (Switching barrier tinggi). Pelanggan tanpa TechSupport cenderung mudah frustrasi dan churn.
8.  `StreamingTV` & `StreamingMovies`.

### C. Informasi Akun (Account Information) - **PALING PENTING**
Biasanya fitur-fitur di sinilah "Training Ground" utama bagi model RF.
9.  `Tenure`: Berapa bulan sudah berlangganan. (Makin lama = makin setia).
10. `Contract`: **Month-to-month**, **One year**, **Two year**.
    *   *Killer Fact*: **Month-to-month** adalah prediktor churn #1. Pelanggan ini bebas pergi kapan saja.
11. `PaperlessBilling`: Tagihan digital vs kertas.
12. `PaymentMethod`: Electronic check, Mailed check, Bank transfer, Credit card.
    *   *Insight*: **Electronic check** sering diasosiasikan dengan churn tinggi.
13. `MonthlyCharges`: Tagihan bulanan.
14. `TotalCharges`: Total uang yang sudah dibayarkan selama menjadi pelanggan (Tenure x Monthly).

**Target Variable**:
*   `Churn`: Yes/No. (Imbalanced: ~26% Yes, ~74% No).

---

## 2. Analisis Mendalam: RF vs KNN (Kenapa hasilnya begitu?)

Berdasarkan hasil run sebelumnya:
*   **Random Forest (RF)**: Menang di **Akurasi & Presisi**.
*   **KNN**: Menang (sedikit) di **Recall**.

### Q: Kenapa Random Forest Lebih Baik Secara Keseluruhan?
*Jawab dengan argumen "High Dimensionality & Karakter Data Campuran".*

1.  **Cara Kerja pada Data Campuran**: Dataset ini punya banyak data Kategorikal (Payment, Contract, Internet) dan Numerik (Tenure).
    *   **RF**: Sangat jago menangani data campuran. Dia membuat keputusan biner ("Apakah kontrak == Month-to-month?"). Ini sangat alami untuk data kategori.
    *   **KNN**: Dia melihat "Jarak". Bagaimana cara menghitung jarak antara "Payment=Check" dan "Payment=Bank"? Kita pakai One-Hot Encoding, yang membuat dimensi data meledak (banyak kolom 0 dan 1). Jarak Euclidean di ruang dimensi tinggi (Sparse Data) menjadi kurang bermakna (*Curse of Dimensionality*). Jarak antara titik A dan B jadi terasa "sama saja".

2.  **Handling Noise/Outlier**:
    *   **RF**: Jika ada data aneh (outlier), pohon keputusan bisa mengisolasi mereka di ranting (leaf) tersendiri tanpa merusak struktur pohon utama.
    *   **KNN**: Sangat sensitif. Jika ada 1 data *churner* aneh (outlier) nyasar di tengah kerumunan pelanggan setia, KNN akan terpengaruh dan bisa salah memprediksi tetangga di sekitarnya.

### Q: Kenapa Recall KNN Lebih Tinggi? (The Recall Paradox)
*Ini poin krusial agar terlihat pinter.*

**Recall** = Kemampuan mendeteksi Churners. "Dari seluruh orang yang aslinya Churn, berapa yang ketangkap?"

1.  **Karakteristik Decision Boundary**:
    *   **KNN**: Membuat batas keputusan yang sangat "bergerigi" dan lokal. Jika di suatu area kecil ada sekumpulan churners, KNN akan mengklaim area itu sebagai "Zona Merah". Ini membuatnya agresif menangkap churners (Recall tinggi), TAPI... dia juga akan menangkap orang tidak bersalah di dekat situ (Banyak False Alarm -> Presisi Rendah).
    *   **Random Forest**: Membangun kotak-kotak keputusan yang lebih besar dan umum (Generalisasi). Untuk menaikkan akurasi global (agar tidak dibilang salah terus), RF cenderung bermain "Aman".
    *   **Faktor Imbalance**: Karena data *No Churn* jauh lebih banyak (74%), RF belajar bahwa menebak "No Churn" itu sering benar. Akibatnya, RF menjadi "konservatif" atau pelit dalam memvonis orang sebagai Churn. Dia hanya akan bilang "Churn" jika buktinya sangat kuat (Presisi tinggi).
    *   **KNN Buta Distribusi**: KNN tidak peduli data global itu 74% No. Dia hanya peduli "5 orang di sebelah saya siapa?". Jika 3 diantaranya Churn, dia teriak Churn. Sifat "Sumbu Pendek" inilah yang membuat Recall-nya tinggi.

---

## 3. Lima Skenario Pertanyaan "Killer" & Jawaban Maut

### Skenario 1: Tentang Preprocessing
**Dosen**: "Saya perhatikan kamu mengisi *missing value* di `TotalCharges` dengan **Mean** (Rata-rata). Kenapa tidak pakai Median? Atau kenapa tidak didrop saja?"

**Jawaban**:
"Saya memilih Mean karena setelah melihat distribusi datanya, nilai `TotalCharges` cukup tersebar tetapi tidak memiliki *extreme outliers* yang merusak nilai rata-rata secara signifikan.
Mengapa tidak di-drop? Karena jumlah datanya hanya 11 baris (0.15% dari data). Meskipun sedikit, setiap data pelanggan di dataset yang terbatas itu berharga.
Namun, untuk *best practice* di kasus real dimana data gaji/biaya sering *skewed* (timpang), **Median** memang biasanya lebih aman. Saya bisa ganti ke Median di iterasi selanjutnya untuk melihat apakah ada perubahan performa, namun dugaan saya efeknya minimal karena jumlah *missing*-nya sangat sedikit."

### Skenario 2: Tentang Metrik Evaluasi
**Dosen**: "Akurasi kamu 80%. Kelihatannya bagus. Tapi kalau saya punya data 90% pelanggan setia, dan model saya cuma menebak 'Setia' terus-terusan, akurasi saya 90%. Jadi akurasi kamu sampah dong?"

**Jawaban**:
"Bapak benar sekali, itulah jebakan *Accuracy Paradox* pada *Imbalanced Dataset*.
Makanya Pak, di presentasi tadi saya **tidak hanya menonjolkan Akurasi**, tapi saya membedah **F1-Score** dan **ROC-AUC**.
*   ROC-AUC saya di atas 0.5 (mendekati 0.8), yang membuktikan model saya punya kemampuan membedakan kelas Positif dan Negatif, bukan sekadar menebak kelas mayoritas.
*   Selain itu, untuk bisnis, kita perlu seimbangkan antara Presisi (efisiensi biaya promo) dan Recall (mencegah pelanggan kabur). Fokus saya adalah menaikkan **Recall** tanpa mengorbankan Presisi terlalu jauh, dan itu tercermin di F1-Score."

### Skenario 3: Algoritma
**Dosen**: "Kenapa Random Forest? Kenapa gak pakai Logistic Regression yang lebih simpel atau XGBoost/Neural Network yang lebih canggih?"

**Jawaban**:
"Saya memilih **Random Forest** sebagai *sweet spot* (jalan tengah):
1.  Dibanding **Logistic Regression**: Regresi Logistik berasumsi hubungan linier. Sedangkan perilaku manusia itu kompleks dan non-linier (misal: orang tenure rendah sering churn, tapi tenure sangat tinggi juga bisa churn karena bosan/pindah rumah). RF bisa menangkap pola non-linier ini.
2.  Dibanding **Neural Network**: Data saya cuma 7000 baris. NN butuh data massive agar tidak overfitting. RF bekerja sangat baik di data ukuran menengah (tabular data) seperti ini.
3.  **XGBoost**: Memang bisa jadi iterasi selanjutnya (future work), tapi Random Forest lebih mudah dijelaskan (*interpretable*) fitur pentingnya ke tim bisnis dibanding metode boosting yang lebih 'black-box'."

### Skenario 4: Hyperparameter Tuning
**Dosen**: "Kamu pakai k=5 untuk KNN. Dari mana angka 5 itu? Wangsit?"

**Jawaban**:
"Angka 5 adalah nilai standar *default* di Scikit-Learn, Pak. Namun, di aplikasi ini saya sudah mengimplementasikan **GridSearchCV** (bisa didemokan di Tab 3).
Fitur ini mencoba berbagai nilai k (misal 3, 5, 7) dan melihat mana yang error-nya paling kecil menggunakan *Cross Validation*.
Jadi nilai parameter bukan tebakan saya, tapi hasil optimasi matematis berdasarkan pola data latih."

### Skenario 5: Business Impact (Paling Killer)
**Dosen**: "Oke modelmu canggih. Tapi kalau model ini salah prediksi (salah tebak), mana yang lebih merugikan perusahaan? False Positive atau False Negative?"

**Jawaban**:
"Pertanyaan bagus Pak. Mari kita hitung:
*   **False Negative (Gagal Deteksi)**: Ada pelanggan mau kabur, kita diamkan. Akibatnya: Kita **Kehilangan Revenue** seumur hidup pelanggan itu (CLV - Customer Lifetime Value). Kerugian: Besar.
*   **False Positive (Salah Tuduh)**: Pelanggan setia, kita kira mau kabur, lalu kita kasih diskon. Akibatnya: Kita keluar **Biaya Diskon** yang tidak perlu. Kerugian: Kecil (biaya marketing).

**Kesimpulan**: Jauh lebih bahaya **False Negative**.
Oleh karena itu, jika saya harus tuning model ini lagi di masa depan, saya akan menggeser *threshold*-nya untuk memprioritaskan **Recall** (menangkap penjahat lebih banyak) meskipun konsekuensinya Presisi turun sedikit (korban salah promo bertambah), karena biaya mendapatkan pelanggan baru 5x lebih mahal daripada mempertahankan yang lama."
