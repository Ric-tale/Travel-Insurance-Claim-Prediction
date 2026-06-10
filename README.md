# Travel Insurance: Claim Prediction

> Proyek Capstone Data Mining — Prediksi Klaim Asuransi Perjalanan menggunakan Machine Learning

---

## Tim

| Nama | 
|------|
| Rico Reswara |
| Alessandro Nicho Hutahaean |
| Daniel Tirta Pratama |
| Muhammad Adiel Faeyza |

---

## Deskripsi Proyek

Proyek ini bertujuan membangun model klasifikasi *Machine Learning* untuk memprediksi apakah seorang pemegang polis asuransi perjalanan akan **mengajukan klaim** atau **tidak** selama perjalanan mereka.

Dengan model ini, perusahaan asuransi dapat:
- Mengelola risiko finansial secara lebih efisien
- Menyesuaikan harga premi berdasarkan profil risiko nasabah
- Mengalokasikan cadangan dana klaim dengan lebih akurat
- Mengidentifikasi faktor-faktor yang paling memengaruhi pengajuan klaim

---

## Target Prediksi

| Label | Kelas | Keterangan |
|-------|-------|------------|
| 0 | No | Nasabah **tidak** mengajukan klaim |
| 1 | Yes | Nasabah **mengajukan** klaim |

> ⚠️ Dataset sangat *imbalanced*: hanya sekitar **1.5%** (~677 dari 44.328 baris) nasabah yang mengajukan klaim.

---

## Struktur Dataset

Dataset berisi informasi historis polis asuransi perjalanan dengan 11 kolom:

| Fitur | Tipe Data | Deskripsi |
|-------|-----------|-----------|
| `Agency` | Kategorikal | Kode agen asuransi (16 agen unik) |
| `Agency Type` | Kategorikal | Jenis agen: *Travel Agency* atau *Airlines* |
| `Distribution Channel` | Kategorikal | Saluran penjualan: *Online* atau *Offline* |
| `Product Name` | Kategorikal | Nama produk asuransi (26 produk unik) |
| `Gender` | Kategorikal | Jenis kelamin nasabah (71.39% *missing*) |
| `Duration` | Numerikal | Durasi perjalanan (hari) |
| `Destination` | Kategorikal | Negara tujuan perjalanan (138 negara unik) |
| `Net Sales` | Numerikal | Nilai premi bersih yang dibayarkan |
| `Commision (in value)` | Numerikal | Nilai komisi agen |
| `Age` | Numerikal | Usia nasabah (tahun) |
| `Claim` | **Target** | 0 = Tidak Klaim, 1 = Klaim |

---

## Alur Kerja (Pipeline)

```
Data Loading
     │
     ▼
Data Cleaning
  ├─ Hapus durasi negatif & ganti ekstrem (> 365 hari) dengan median
  ├─ Hapus refund dengan status klaim = 1 & Net Sales negatif
  └─ Ganti usia > 100 tahun dengan median
     │
     ▼
Exploratory Data Analysis (EDA)
  ├─ Distribusi kelas target (class imbalance)
  ├─ Distribusi fitur numerik berdasarkan status klaim
  ├─ Claim rate per fitur kategorikal
  ├─ Top 10 Destination & Product Name berdasarkan claim rate
  └─ Heatmap korelasi fitur numerik
     │
     ▼
Preprocessing & Feature Engineering
  ├─ Imputasi Gender (missing value → 'Unknown')
  ├─ Binary Encoding (fitur kategorikal high-cardinality)
  └─ RobustScaler (fitur numerik, tahan outlier)
     │
     ▼
Modeling & Cross Validation (5-Fold Stratified)
  ├─ Logistic Regression ✅ (Terbaik)
  ├─ Decision Tree
  ├─ K-Nearest Neighbors
  ├─ Random Forest
  ├─ XGBoost
  └─ LightGBM
     │
     ▼
Hyperparameter Tuning (GridSearchCV)
     │
     ▼
Evaluasi Model (Test Set)
  ├─ ROC-AUC Score
  ├─ Recall, Precision, F1-Score
  └─ Confusion Matrix
     │
     ▼
Feature Importance & Simpan Model (.pkl)
```

---

## Hasil Model

**Model Terbaik: Logistic Regression**

| Metrik | Cross Validation | Test Set |
|--------|-----------------|----------|
| ROC-AUC | 0.8087 | **0.83** |
| Recall (Kelas 1) | 0.7011 | ~0.69 |

Logistic Regression unggul karena:
1. Data sangat *imbalanced* (rasio ~98:2) — model linear lebih stabil dalam kondisi ini
2. Fitur numerik utama (*Duration*, *Age*, *Net Sales*) memiliki hubungan linear yang kuat dengan target
3. Lebih tahan *overfitting* dibanding model berbasis pohon pada kondisi data ekstrem

---

## Fitur Paling Berpengaruh

1. **Duration** — Perjalanan lebih lama → eksposur risiko lebih tinggi
2. **Net Sales** — Premi lebih tinggi berkorelasi dengan cakupan pertanggungan yang lebih luas
3. **Age** — Nasabah lebih tua umumnya memiliki risiko kesehatan lebih tinggi
4. **Commision** — Berkorelasi dengan jenis produk dan profil agen

---

## Instalasi & Dependensi

### Prasyarat
- Python 3.8+
- Jupyter Notebook / JupyterLab

### Instalasi Library

```bash
pip install pandas numpy matplotlib seaborn missingno scikit-learn
pip install category_encoders imbalanced-learn xgboost lightgbm
```

Atau menggunakan satu perintah:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
pandas
numpy
matplotlib
seaborn
missingno
scikit-learn
category_encoders
imbalanced-learn
xgboost
lightgbm
```

---

## Cara Menjalankan

1. **Clone repository ini**
   ```bash
   git clone <url-repo>
   cd travel-insurance-claim-prediction
   ```

2. **Siapkan dataset**
   Letakkan file `data_travel_insurance.csv` di direktori yang sama dengan notebook.

3. **Jalankan notebook**
   ```bash
   jupyter notebook CapstoneDataMining_TravelInsurance_FINAL.ipynb
   ```

4. **Jalankan sel secara berurutan** dari atas ke bawah.

---

## Menggunakan Model yang Tersimpan

Setelah training, model disimpan sebagai file pickle. Gunakan kode berikut untuk memuat dan melakukan prediksi:

```python
import pickle
import pandas as pd

# Load model
with open('best_model_travel_insurance.pkl', 'rb') as f:
    model = pickle.load(f)

# Contoh data baru (sesuaikan dengan kolom yang dibutuhkan)
data_baru = pd.DataFrame([{
    'Agency': 'EPX',
    'Agency Type': 'Travel Agency',
    'Distribution Channel': 'Online',
    'Product Name': 'Cancellation Plan',
    'Gender': 'M',
    'Duration': 30,
    'Destination': 'SINGAPORE',
    'Net Sales': 26.0,
    'Commision (in value)': 0.0,
    'Age': 35
}])

# Prediksi probabilitas klaim
probabilitas = model.predict_proba(data_baru)[:, 1]
print(f"Probabilitas Klaim: {probabilitas[0]:.4f}")
```

---

## Limitasi Model

1. **Missing Value Gender (71.39%)** — Kontribusi fitur Gender terbatas; tidak dapat diandalkan untuk analisis demografis
2. **Class Imbalance Ekstrem (98:2)** — Meski ditangani dengan RandomOverSampler, Precision kelas klaim tetap sangat rendah (~0.06)
3. **Data Historis** — Model dapat menurun performanya jika terjadi perubahan mendadak pola nasabah, kondisi geopolitik, atau *force majeure*
4. **Batasan Operasional:**
   - Andal untuk: nasabah usia 18–80 tahun, durasi < 365 hari, destinasi umum
   - Perlu hati-hati: durasi > 1000 hari, usia ekstrem, destinasi sangat jarang
5. **Tidak ada analisis temporal** — belum mempertimbangkan tren perubahan pola klaim dari waktu ke waktu

---

## Rekomendasi Bisnis

1. **Dynamic Pricing** — Terapkan premi dinamis berbasis skor probabilitas klaim dari model
2. **Efisiensi Cadangan Dana** — Gunakan prediksi model untuk memproyeksikan estimasi rasio klaim per periode
3. **Evaluasi Keagenan** — Tinjau ulang agen dengan *claim rate* tinggi dan perketat *underwriting* untuk destinasi berisiko
4. **Roadmap Model:**
   - Lengkapi data Gender untuk mengurangi *missing value*
   - Coba *cost-sensitive learning* atau teknik *undersampling* untuk meningkatkan Precision
   - Integrasikan variabel eksternal (indeks risiko keamanan, indeks kesehatan negara tujuan)
   - Jadwalkan *re-training* berkala dengan data terbaru

---

## Metrik Evaluasi

| Metrik | Keterangan |
|--------|------------|
| **ROC-AUC** | Metrik utama — mengukur kemampuan diskriminasi model di semua threshold; tidak terpengaruh *class imbalance* |
| **Recall** | Penting untuk meminimalkan *False Negative* (klaim tidak terdeteksi → kerugian finansial mendadak) |
| **Precision** | Mengukur akurasi prediksi positif (*False Positive* → dana cadangan tidak perlu) |
| **F1-Score** | Harmonic mean Precision & Recall sebagai ringkasan keseimbangan kedua metrik |

---
