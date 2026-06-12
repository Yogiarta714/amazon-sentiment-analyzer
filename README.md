# 🍲 Amazon Fine Food Sentiment Analyzer

> Aplikasi web berbasis kecerdasan buatan untuk mengklasifikasikan sentimen ulasan produk makanan dan minuman menggunakan **Logistic Regression** dan pembobotan kata **TF-IDF**.

---

## 📋 Daftar Isi

- [Deskripsi Proyek](#-deskripsi-proyek)
- [Tujuan](#-tujuan)
- [Demo Aplikasi](#-demo-aplikasi)
- [Dataset](#-dataset)
- [Arsitektur & Pipeline](#-arsitektur--pipeline)
- [Struktur Direktori](#-struktur-direktori)
- [Teknologi & Library](#-teknologi--library)
- [Cara Instalasi](#-cara-instalasi)
- [Cara Menjalankan Aplikasi](#-cara-menjalankan-aplikasi)
- [Cara Penggunaan](#-cara-penggunaan)
- [Penjelasan Notebook (Google Colab)](#-penjelasan-notebook-google-colab)
- [Penjelasan Pipeline Preprocessing](#-penjelasan-pipeline-preprocessing)
- [Penjelasan Model Machine Learning](#-penjelasan-model-machine-learning)
- [Detail File](#-detail-file)

---

## 📖 Deskripsi Proyek

**Amazon Fine Food Sentiment Analyzer** adalah aplikasi analisis sentimen berbasis web yang dibangun menggunakan framework **Streamlit**. Aplikasi ini mampu memprediksi apakah sebuah ulasan produk makanan atau minuman berbahasa Inggris bersifat **positif (pujian/rekomendasi)** atau **negatif (keluhan/kekecewaan)**.

Proyek ini terdiri dari dua bagian utama:
1. **Notebook pelatihan** (`sentiment-analyzer.ipynb`) : proses eksplorasi data, preprocessing, pelatihan model, dan evaluasi yang dijalankan di Google Colab.
2. **Web App Streamlit** (`app.py`) : antarmuka pengguna interaktif yang menggunakan model hasil pelatihan untuk melakukan prediksi secara real-time.

Model dilatih menggunakan algoritma **Logistic Regression** dengan representasi teks berbasis **TF-IDF (Term Frequency-Inverse Document Frequency)**, dan dilakukan eksperimen perbandingan antara model **dengan stopword** dan **tanpa stopword** untuk menemukan konfigurasi terbaik.

---

## 🎯 Tujuan

1. **Mengklasifikasikan sentimen** ulasan makanan/minuman secara otomatis (positif atau negatif).
2. **Membandingkan pengaruh stopword removal** terhadap performa model Logistic Regression.
3. **Menerapkan pipeline NLP** lengkap mulai dari data mentah hingga prediksi model.
4. **Memberikan transparansi** kepada pengguna dengan menampilkan setiap tahapan pemrosesan teks secara detail (Pipeline Log).
5. **Menyediakan antarmuka yang ramah pengguna** melalui web app interaktif berbasis Streamlit.

---

## 🖥️ Demo Aplikasi

Masukkan teks ulasan berbahasa Inggris, lalu tekan tombol **"Analisis Sentimen Ulasan"**. Contoh input:

```
This coffee tastes amazing! It has a rich flavor and the packaging was perfect.
```

Aplikasi akan menampilkan:
- Label sentimen (**POSITIF** atau **NEGATIF**)
- Tingkat keyakinan model dalam bentuk persentase
- Detail pipeline pemrosesan internal (opsional, dapat dibuka via expander)

---

## 📊 Dataset

| Atribut | Detail |
|---|---|
| **Nama** | Amazon Fine Food Reviews |
| **Sumber** | [Kaggle — SNAP Stanford](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) |
| **File** | `Reviews.csv` |
| **Jumlah Data** | ± 568.454 ulasan |
| **Rentang Waktu** | Oktober 1999 – Oktober 2012 |
| **Bahasa** | Inggris |

### Kolom yang Digunakan

| Kolom | Deskripsi |
|---|---|
| `Score` | Rating produk (1–5 bintang) |
| `Text` | Isi teks ulasan dari pengguna |
| `label` | Label sentimen hasil proses labeling (0 = Negatif, 1 = Positif) |

### Skema Labeling

Rating `Score` dikonversi menjadi label biner dengan aturan berikut:

| Score | Label | Keterangan |
|---|---|---|
| 1 – 2 | `0` | Sentimen **Negatif** |
| 3 | *(dibuang)* | Netral — tidak digunakan dalam pelatihan |
| 4 – 5 | `1` | Sentimen **Positif** |

> Rating 3 (netral) sengaja dihapus dari dataset karena tidak memberikan sinyal sentimen yang jelas, sehingga tidak representatif untuk klasifikasi biner.

---

## ⚙️ Arsitektur & Pipeline

### Pipeline Pelatihan (Notebook)

```
Reviews.csv (Dataset Mentah)
        │
        ▼
1. Labeling             → Score 1-2 = Negatif (0), Score 4-5 = Positif (1)
                          Score 3 dibuang
        │
        ▼
2. Case Folding         → Konversi teks ke huruf kecil
        │
        ▼
3. Text Cleaning        → Hapus URL, mention, hashtag, angka, tanda baca
        │
        ▼
4. Tokenisasi           → Pecah teks menjadi daftar token/kata
        │
        ▼
5. Stopword Removal     → Buat 2 versi: dengan & tanpa stopword
                          (kata "not" dipertahankan di kedua versi)
        │
        ▼
6. Train-Test Split     → 80% data latih, 20% data uji
        │
        ▼
7. TF-IDF Vectorization → Representasi numerik teks (fit pada data latih)
        │
        ▼
8. Logistic Regression  → Latih 2 model (dengan & tanpa stopword)
        │
        ▼
9. Evaluasi             → Akurasi, Classification Report, Confusion Matrix,
                          Analisis Bobot Kata
        │
        ▼
10. Simpan Model Terbaik → logistic_regression_model.pickle
                           tfidf_vectorizer.pickle
```

### Pipeline Inferensi (Web App)

```
Input Teks Pengguna
        │
        ▼
1. Case Folding  →  2. Cleaning  →  3. Tokenisasi  →  4. Stopword Removal
        │
        ▼
5. TF-IDF Transform  →  6. Logistic Regression  →  Output Prediksi
```

---

## 📁 Struktur Direktori

```
📦 amazon-sentiment-analyzer/
 ┣ 📓 sentiment-analyzer.ipynb         # Notebook Google Colab: EDA, preprocessing, training, evaluasi
 ┣ 📄 requirements.txt                 # Daftar dependensi library Python
 ┣ 📄 README.md                        # Dokumentasi proyek
 ┗ 📦 sentiment-analyzer-app/
    ┣ 📄 app.py                            # File utama aplikasi web Streamlit
    ┣ 📄 preprocessing.py                  # Modul fungsi preprocessing teks (reusable)
    ┣ 📄 logistic_regression_model.pickle  # Model Logistic Regression hasil training (pre-trained)
    ┗ 📄 tfidf_vectorizer.pickle           # Objek TF-IDF Vectorizer hasil fitting (pre-trained)
```

> **Catatan:** File `.pickle` merupakan model yang sudah dilatih sebelumnya. File ini wajib tersedia di dalam folder `sentiment-analyzer-app/` agar aplikasi dapat berjalan. Dataset `Reviews.csv` **tidak disertakan** dalam repositori karena ukurannya yang besar, unduh secara mandiri dari Kaggle (lihat bagian [Dataset](#-dataset)).

---

## 🛠️ Teknologi & Library

### Library Aplikasi Web (Streamlit)

| Library | Kegunaan |
|---|---|
| `streamlit` | Framework web app interaktif berbasis Python |
| `scikit-learn` | Logistic Regression & TF-IDF Vectorizer |
| `nltk` | Tokenisasi & stopword removal |
| `numpy` | Operasi array numerik |
| `pickle` *(built-in)* | Serialisasi dan deserialisasi model ML |
| `re` *(built-in)* | Regular expression untuk text cleaning |

### Library Tambahan (Notebook / Training)

| Library | Kegunaan |
|---|---|
| `pandas` | Manipulasi dan analisis dataframe |
| `matplotlib` | Visualisasi grafik dan plot |
| `seaborn` | Visualisasi heatmap confusion matrix |
| `google.colab` | Mount Google Drive di lingkungan Colab |

### Resource NLTK (Diunduh Otomatis)

| Resource | Kegunaan |
|---|---|
| `punkt` | Tokenizer berbasis bahasa |
| `punkt_tab` | Tokenizer tabel (versi terbaru NLTK) |
| `stopwords` | Daftar kata umum bahasa Inggris |

---

## 🚀 Cara Instalasi

### Prasyarat

- Python **3.8** atau lebih baru
- `pip` (Python package manager)
- (Opsional) Virtual environment seperti `venv` atau `conda`

### Langkah Instalasi

**1. Clone repositori ini**

```bash
git clone https://github.com/Yogiarta714/amazon-sentiment-analyzer.git
cd amazon-sentiment-analyzer
```

**2. Buat dan aktifkan virtual environment** (Opsional, tetapi sangat disarankan)

```bash
# Menggunakan venv
python -m venv venv

# Aktivasi (Windows)
venv\Scripts\activate

# Aktivasi (macOS / Linux)
source venv/bin/activate
```

**3. Install seluruh dependensi**

```bash
pip install -r requirements.txt
```

**4. Pastikan file model tersedia**

Pastikan kedua file berikut sudah ada di dalam folder `sentiment-analyzer-app/`:
- `logistic_regression_model.pickle`
- `tfidf_vectorizer.pickle`

> Resource NLTK (`punkt`, `stopwords`) akan **diunduh otomatis** saat aplikasi pertama kali dijalankan.

---

## ▶️ Cara Menjalankan Aplikasi

Setelah instalasi selesai, masuk ke folder aplikasi lalu jalankan Streamlit:

```bash
cd sentiment-analyzer-app
streamlit run app.py
```

Aplikasi akan otomatis terbuka di browser pada alamat:

```
http://localhost:8501
```

---

## 📌 Cara Penggunaan

1. Buka aplikasi di browser (`http://localhost:8501`).
2. Ketikkan atau tempel teks ulasan makanan/minuman **berbahasa Inggris** pada kolom input.
3. Klik tombol **"Analisis Sentimen Ulasan"**.
4. Hasil prediksi akan muncul:
   - ✅ **Hijau** → Sentimen **POSITIF** (pujian/rekomendasi)
   - ❌ **Merah** → Sentimen **NEGATIF** (keluhan/kekecewaan)
   - Disertai **persentase tingkat keyakinan** model.
5. (Opsional) Klik **"🔍 Lihat Detail Pemrosesan Internal Model (Pipeline Log)"** untuk melihat hasil tiap tahap preprocessing dan nilai bobot TF-IDF yang terdeteksi.

---

## 📓 Penjelasan Notebook (Google Colab)

File `sentiment-analyzer.ipynb` berisi seluruh proses riset dan pelatihan model dari awal hingga akhir. Notebook ini dijalankan di **Google Colab** dengan data bersumber dari Google Drive.

Notebook dibagi menjadi empat bagian utama:

### 1. Preparation
- Import seluruh library yang dibutuhkan (`pandas`, `re`, `nltk`, `matplotlib`, `seaborn`)
- Mount Google Drive untuk mengakses file `Reviews.csv`
- Load dataset dan eksplorasi awal: jumlah baris, nama kolom, distribusi rating, dan contoh data

### 2. Preprocessing
Teks diproses melalui lima tahap secara berurutan:

| Tahap | Fungsi | Keterangan |
|---|---|---|
| Labeling | `tentukan_sentimen()` | Konversi Score → label 0/1, buang rating 3 |
| Case Folding | `.str.lower()` | Semua huruf menjadi lowercase |
| Cleaning | `clean_text()` | Hapus URL, mention, hashtag, angka, tanda baca |
| Tokenisasi | `word_tokenize()` | Pecah kalimat menjadi list token |
| Stopword Removal | `buang_stopword()` | Filter stopword, pertahankan kata "not" |

Hasil preprocessing menghasilkan **dua versi kolom teks**:
- `final_with_stopwords` : teks yang masih mengandung stopword (stopword tidak dihapus)
- `final_no_stopwords`   : teks yang stopword-nya sudah dihapus

### 3. Representasi TF-IDF
- Data dibagi menjadi **80% training** dan **20% testing** menggunakan `train_test_split` dengan `random_state=42`
- `TfidfVectorizer` di-*fit* pada data training dan digunakan untuk mentransformasi data training maupun testing
- Dua vectorizer dibuat secara terpisah untuk masing-masing versi (dengan/tanpa stopword)
- Contoh representasi TF-IDF untuk satu review ditampilkan secara detail (kata + nilai bobot)

### 4. Machine Learning — Logistic Regression
- **Pelatihan:** Dua model `LogisticRegression(max_iter=1000)` dilatih secara terpisah
- **Evaluasi:** Akurasi, `classification_report` (precision, recall, F1-score), dan confusion matrix ditampilkan untuk kedua model
- **Analisis Bobot:** 10 kata dengan bobot tertinggi (mendorong ke sentimen positif) dan 10 kata dengan bobot terendah (mendorong ke sentimen negatif) dianalisis untuk interpretasi model

---

## 🔬 Penjelasan Pipeline Preprocessing

Pipeline preprocessing diimplementasikan dalam `preprocessing.py` dan terdiri dari tahap berikut:

### 1. Case Folding
Mengubah seluruh karakter teks menjadi huruf kecil agar kata yang sama dengan kapitalisasi berbeda dianggap identik.
```
"This Coffee is GREAT" → "this coffee is great"
```

### 2. Text Cleaning (`clean_text`)
Membersihkan teks dari noise menggunakan Regular Expression (Regex):
- Menghapus URL/link (`http://...`)
- Menghapus mention (`@username`)
- Menghapus simbol hashtag (`#`)
- Menghapus angka dan tanda baca (hanya menyisakan huruf a–z dan spasi)
- Menghapus spasi berlebih

```
"Check http://amzn.com @seller #review! 5 stars!!" → "Check seller review stars"
```

### 3. Tokenisasi (`word_tokenize`)
Memecah string teks menjadi daftar token (kata) menggunakan NLTK `word_tokenize`.
```
"this coffee is great" → ["this", "coffee", "is", "great"]
```

### 4. Stopword Removal (`buang_stopword`)
Menghapus kata-kata umum yang tidak membawa makna signifikan menggunakan daftar stopword bahasa Inggris dari NLTK.

> ⚠️ **Perhatian khusus:** Kata **"not"** sengaja **dipertahankan** karena sangat krusial dalam menentukan polaritas sentimen. Contoh: *"not good"* berbeda makna dengan *"good"*.

```
["this", "coffee", "is", "great"]       → ["coffee", "great"]
["this", "product", "is", "not", "good"] → ["product", "not", "good"]
```

---

## 🤖 Penjelasan Model Machine Learning

### TF-IDF Vectorizer
**TF-IDF (Term Frequency-Inverse Document Frequency)** mengubah teks menjadi representasi vektor numerik. Nilai TF-IDF tinggi menunjukkan kata tersebut sering muncul di dokumen tertentu namun jarang di dokumen lain, artinya kata tersebut lebih informatif dan relevan.

- **TF (Term Frequency):** seberapa sering suatu kata muncul dalam satu dokumen
- **IDF (Inverse Document Frequency):** penalti untuk kata yang muncul di hampir semua dokumen

### Logistic Regression
Model klasifikasi biner yang memprediksi probabilitas sentimen. Konfigurasi yang digunakan: `LogisticRegression(max_iter=1000)`.

| Probabilitas | Label | Output |
|---|---|---|
| > 0.5 | Positif (1) | Pujian / Rekomendasi |
| ≤ 0.5 | Negatif (0) | Keluhan / Kekecewaan |

Model dan vectorizer disimpan dalam format `.pickle` sehingga dapat digunakan langsung oleh aplikasi Streamlit tanpa perlu melatih ulang.

### Eksperimen Perbandingan

Notebook melakukan eksperimen untuk membandingkan dua konfigurasi:

| Konfigurasi | Deskripsi |
|---|---|
| Model A | Logistic Regression + TF-IDF **dengan** stopword |
| Model B | Logistic Regression + TF-IDF **tanpa** stopword |

Hasil evaluasi kedua model dibandingkan menggunakan akurasi, precision, recall, F1-score, dan confusion matrix untuk menentukan model mana yang lebih baik dan akhirnya disimpan sebagai model produksi.

---

## 📄 Detail File

### `sentiment-analyzer.ipynb`
Notebook Google Colab yang berisi seluruh proses dari awal hingga model siap digunakan: eksplorasi data, labeling, preprocessing, splitting, TF-IDF, pelatihan, evaluasi, analisis bobot kata, dan penyimpanan model ke file `.pickle`.

### `app.py`
File utama aplikasi Streamlit. Bertanggung jawab atas konfigurasi halaman, pemuatan model dari file `.pickle`, penerimaan input pengguna, pemanggilan fungsi preprocessing, prediksi, dan penampilan hasil beserta pipeline log.

### `preprocessing.py`
Modul Python reusable yang berisi fungsi-fungsi preprocessing teks:
- `download_nltk_resources()` : mengunduh resource NLTK secara otomatis jika belum tersedia
- `clean_text(text)` : membersihkan teks dari noise menggunakan regex
- `word_tokenize(text)` : tokenisasi teks (diimpor dari NLTK)
- `buang_stopword(tokens)` : menghapus stopword kecuali kata "not"

### `logistic_regression_model.pickle`
File biner hasil serialisasi objek model Logistic Regression yang sudah dilatih. Dimuat menggunakan `pickle.load()` saat aplikasi pertama kali dijalankan.

### `tfidf_vectorizer.pickle`
File biner hasil serialisasi objek TF-IDF Vectorizer yang sudah di-*fit* pada data training. Digunakan untuk mentransformasi teks baru menjadi vektor numerik yang konsisten dengan representasi saat pelatihan.

### `requirements.txt`
Daftar seluruh library Python beserta versinya yang dibutuhkan untuk menjalankan aplikasi. Install menggunakan `pip install -r requirements.txt`.

---

<p align="center">
  Dibuat dengan ❤️ menggunakan Python & Streamlit
</p>
