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
