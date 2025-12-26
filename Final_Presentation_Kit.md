# Kit Validasi & Presentasi Final (All-in-One)

Dokumen ini menggabungkan **Naskah Presentasi** (apa yang diucapkan) dan **Panduan Belajar** (pemahaman konsep) dalam bahasa yang santai dan mudah dimengerti. Anggap ini contekan pintar Anda.

---

# BAGIAN 1: NASKAH PRESENTASI (The Show)

Ikuti alur ini saat Anda berdiri di depan kelas sambil menjalankan aplikasi Streamlit.

## 1. Pembukaan (Opening)
**Ucapkan:**
"Halo semua. Bayangkan Anda punya bisnis langganan internet. Pasti sakit rasanya kalau pelanggan lama tiba-tiba berhenti (Churn) dan pindah ke kompetitor.
Hari ini, saya membuat aplikasi cerdas yang bisa:
1.  Menemukan **siapa** yang mau kabur.
2.  Tahu **kenapa** mereka kabur.
3.  Memberi **saran** agar mereka tetap setia.
Mari kita lihat demonya."

## 2. Menu "Data Exploration" (Kenalan dengan Data)
**Aksi di Layar:** Tunjukkan Pie Chart `Churn` dan grafik `PaymentMethod`.

**Ucapkan:**
"Pertama, kita cek dulu datanya.
*   Lihat Pie Chart ini? Sebagian besar pelanggan masih setia (Warna Biru), dan cuma sedikit yang kabur (Merah). Ini wajar di dunia nyata, tapi bikin komputer susah belajar karena contoh 'orang kabur'-nya sedikit (Data Imbalance).
*   Yang menarik, coba lihat grafik kanan. Pelanggan yang bayar pakai **Electronic Check** itu paling banyak yang kabur. Mungkin karena sistemnya ribet atau manual?"

## 3. Menu "Preprocessing" (Bersih-bersih Data)
**Aksi di Layar:** Klik tombol `Run Preprocessing` dan tunjukkan beda tabel Before-After.

**Ucapkan:**
"Data mentah itu ibarat bahan makanan yang belum dicuci. Sebelum dimasak oleh AI, kita harus olah dulu:
1.  **Deterjemahkan**: Komputer gak ngerti huruf. Jadi kata 'Pria/Wanita' kita ubah jadi angka 0 dan 1.
2.  **Disamakan**: Gaji ribuan dollar dan masa langganan (bulan) itu beda skala. Kita samakan skalanya biar adil.
Hasilnya adalah tabel angka di sebelah kanan yang siap dipelajari mesin."

## 4. Menu "Model Training" (Otak AI Belajar)
**Aksi di Layar:** Klik `Start Training`. Tunggu loading selesai.

**Ucapkan:**
"Di sini saya mengadu dua otak kecerdasan buatan:
1.  **Random Forest**: Ibarat sekumpulan pakar yang berdiskusi (voting) untuk mengambil keputusan.
2.  **KNN (Tetangga Terdekat)**: Ibarat orang yang ikut-ikutan teman sebelahnya. Kalau teman-temannya kabur, dia ikutan kabur.
Saya juga pakai fitur 'GridSearch' biar komputer otomatis mencari settingan paling optimal."

## 5. Menu "Results" (Siapa Pemenangnya?)
**Aksi di Layar:** Fokus ke Confusion Matrix atau Bar Chart.

**Ucapkan:**
"Ternyata, **Random Forest menang** secara keseluruhan. Dia lebih pintar membedakan mana pelanggan setia dan mana yang bukan.
Tapi lihat grafik **Feature Importance** ini (tunjukkan grafik batang).
Ternyata penyebab utama orang kabur adalah:
1.  **Tagihan Bulanan**: Kemahalan.
2.  **Tenure**: Pelanggan baru lebih labil daripada pelanggan lama.
3.  **Kontrak**: Yang kontraknya bulanan (bebas pergi) lebih gampang kabur dibanding yang kontrak tahunan."

## 6. Menu "Business Insight" (Simulasi Nyata)
**Aksi di Layar:** Masukkan data simulasi (Tenure rendah, Tagihan tinggi, Kontrak Bulanan) -> Klik Predict.

**Ucapkan:**
"Sekarang kita coba simulasi. Ada pelanggan baru, tagihan mahal, kontrak bulanan.
*Klik Predict*
Nah! Sistem langsung deteksi **Bahaya (High Risk)**.
Gak cuma deteksi, sistem langsung kasih saran: 'Tawarkan diskon kalau dia mau ambil kontrak 1 tahun'. Solutif kan?"

---

# BAGIAN 2: JAWABAN PERTANYAAN DOSEN (The Defense)

Bagian ini adalah "bocoran jawaban" kalau dosen killer mulai bertanya aneh-aneh. Penjelasannya pakai logika sederhana.

### Q1: "Kenapa Random Forest menang lawan KNN?"
**Jawab Santai:**
"Karena data kita ini campur aduk, Pak. Ada data kategori (jenis internet, cara bayar) dan data angka (duit, bulan).
*   **KNN** itu bingung kalau disuruh ngukur jarak antara 'Bayar Pakai Cek' dan 'Transfer Bank'. Jaraknya berapa meter? Gak jelas kan.
*   **Random Forest** kerjanya kayak kuis Yes/No. 'Apakah dia bayar pakai cek? Ya. Apakah tagihan mahal? Ya.' Jadi dia lebih jago menangani data gado-gado kayak gini."

### Q2: "Recall itu apa sih? Kenapa penting?"
**Jawab Santai:**
"Recall itu kemampuan 'Kepekaan' atau daya tangkap maling, Pak.
Misal ada 10 pelanggan yang aslinya mau kabur.
*   Kalau **Recall tinggi**: Kita berhasil tangkap 9 orang. (Bagus, kita selamatkan aset).
*   Kalau **Recall rendah**: Kita cuma sadar 2 orang. Sisanya 8 orang diam-diam kabur tanpa kita tahu.
Di bisnis ini, kita butuh Recall lumayan tinggi biar gak kecolongan."

### Q3: "Kalau modelmu salah tebak, ruginya apa?"
**Jawab Santai:**
"Ada dua jenis salah tebak, Pak:
1.  **Salah Tuduh (False Positive)**: Orang setia dituduh mau kabur.
    *   *Akibat*: Kita kasih dia diskon.
    *   *Rugi*: Sedikit (cuma rugi harga diskon).
2.  **Gagal Deteksi (False Negative)**: Orang mau kabur dikira setia.
    *   *Akibat*: Kita diamkan, eh besoknya dia putus langganan.
    *   *Rugi*: **BESAR**. Kita kehilangan pemasukan dari dia selamanya.

Jadi, tujuan saya bikin aplikasi ini adalah meminimalisir **Gagal Deteksi** itu."

### Q4: "Missing Value di TotalCharges kenapa diisi Rata-rata?"
**Jawab Santai:**
"Karena yang kosong cuma sedikit banget (cuma 11 orang dari 7000). Jadi kalau saya isi pakai nilai rata-rata, tidak akan merusak data keseluruhan. Ibarat seember air tawar, dituang satu sendok teh gula, rasanya gak berubah manis kan?"

### Q5: "Apa itu GridSearch?"
**Jawab Santai:**
"Itu cara komputer nyobain semua kunci jawaban satu-satu, Pak.
Daripada saya menebak-nebak settingan model (misal jumlah pohonnya 50 atau 100?), saya suruh komputer coba semuanya, lalu lapor ke saya mana yang nilainya paling bagus. Jadi hasilnya pasti optimal, bukan tebakan."

---

# TIPS TAMBAHAN BIAR KELIHATAN PINTAR

1.  **Jangan cuma baca slide**. Tatap mata dosen saat menjelaskan "Bagian 2".
2.  **Demonstrasi itu Kunci**. Habiskan waktu lebih banyak di Menu 5 (Business Insight). Dosen suka melihat *hasil nyata* dibanding kode rumit.
3.  **Jujur**. Kalau ditanya dan gak tahu, bilang: "Wah menarik Pak, saya belum riset sedalam itu, tapi logika saya mengatakan..." (lalu pakai logika di atas).

Semoga sukses sidangnya! 🚀
