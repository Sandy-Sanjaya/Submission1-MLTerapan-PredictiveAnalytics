# Laporan Proyek Machine Learning - Sandy Sanjaya

## Domain Proyek
Energi merupakan salah satu kebutuhan vital dalam kehidupan modern. Dalam konteks bangunan, baik komersial maupun residensial, konsumsi energi yang efisien menjadi tantangan tersendiri. Salah satu cara untuk menjawab tantangan tersebut adalah dengan menerapkan pendekatan berbasis data, khususnya menggunakan teknik predictive analytics untuk memprediksi konsumsi energi berdasarkan parameter lingkungan, operasional, dan perilaku pengguna.

Dengan melakukan prediksi terhadap konsumsi energi, institusi atau perusahaan dapat mengatur penggunaan sistem HVAC (Heating, Ventilation, and Air Conditioning), pencahayaan, hingga integrasi energi terbarukan secara lebih optimal. Hal ini berkontribusi terhadap penghematan biaya, pengurangan jejak karbon, dan peningkatan efisiensi energi secara keseluruhan.

Menurut laporan International Energy Agency (IEA), bangunan menyumbang hampir 30% dari konsumsi energi global dan sekitar 28% dari emisi karbon dioksida terkait energi (IEA, 2023). Oleh karena itu, penting untuk mengembangkan sistem prediktif yang dapat membantu pengelolaan energi secara cerdas dan berbasis data.

Dalam proyek ini, dilakukan eksplorasi terhadap berbagai fitur seperti suhu, kelembapan, luas bangunan, jumlah orang dalam bangunan (occupancy), dan penggunaan energi terbarukan untuk memprediksi total konsumsi energi (Energy Consumption). Data yang digunakan mencakup waktu (bulan, jam, dan hari), serta status hari libur, sehingga dimungkinkan untuk menangkap pola-pola temporal yang relevan terhadap perilaku konsumsi energi.

Referensi: 
* International Energy Agency. (2023). Tracking Buildings 2023. Retrieved from: https://www.iea.org/reports/tracking-buildings-2023
* Amasyali, K., & El-Gohary, N. M. (2018). A review of data-driven building energy consumption prediction studies. Renewable and Sustainable Energy Reviews, 81, 1192–1205. https://doi.org/10.1016/j.rser.2017.04.095
* Zhao, H. X., & Magoulès, F. (2012). A review on the prediction of building energy consumption. Renewable and Sustainable Energy Reviews, 16(6), 3586–3592. https://doi.org/10.1016/j.rser.2012.02.049

## Business Understanding

### Problem Statements
- Bagaimana cara memprediksi konsumsi energi bangunan berdasarkan variabel lingkungan dan operasional seperti suhu, kelembapan, waktu, dan penggunaan HVAC?
Banyak organisasi kesulitan dalam memperkirakan kebutuhan energi secara akurat, yang menyebabkan pemborosan energi atau bahkan kekurangan pasokan.
- Apakah ada hubungan signifikan antara waktu (bulan, hari, dan jam) serta faktor internal bangunan (occupancy dan square footage) terhadap tingkat konsumsi energi?
Tanpa pemahaman mendalam mengenai pola konsumsi energi berdasarkan waktu dan karakteristik bangunan, pengelolaan energi tidak dapat dilakukan secara efisien.
- Bagaimana kontribusi energi terbarukan terhadap konsumsi total energi, dan dapatkah model memprediksi energi bersih (net energy consumption) secara lebih akurat?
Dengan meningkatnya adopsi energi terbarukan, penting untuk mengidentifikasi sejauh mana energi bersih berkontribusi dalam menekan konsumsi dari sumber konvensional.

### Goals
- Menghasilkan model prediksi konsumsi energi yang mampu memberikan estimasi akurat berdasarkan data historis dan parameter lingkungan serta operasional. 
- Menganalisis variabel-variabel signifikan yang mempengaruhi konsumsi energi, sehingga dapat dilakukan optimasi terhadap parameter yang paling berpengaruh seperti suhu, occupancy, dan jam penggunaan.
- Mengukur dampak energi terbarukan terhadap total konsumsi energi, sebagai bagian dari upaya transisi menuju bangunan yang lebih ramah lingkungan dan efisien energi.

### Solution Statements 
Untuk menjawab problem statements dan mencapai goals yang telah dijabarkan, solusi berikut diterapkan dalam proyek ini:
1. Menggunakan enam algoritma regresi untuk membandingkan performa dan mendapatkan model terbaik, yaitu:
- Linear Regression: sebagai baseline model yang sederhana namun interpretatif.
- Ridge Regression: untuk menangani multikolinearitas antar fitur melalui regularisasi L2.
- Lasso Regression: untuk melakukan seleksi fitur melalui regularisasi L1.
- Random Forest Regressor: untuk menangani relasi non-linear dan interaksi kompleks antar fitur.
- Gradient Boosting Regressor: untuk membangun model kuat secara bertahap dari kesalahan model sebelumnya.
- XGBoost Regressor: sebagai versi penyempurnaan dari Gradient Boosting dengan optimasi kecepatan dan performa.
2. Membandingkan performa model menggunakan metrik regresi, yaitu:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

Evaluasi ini digunakan untuk menilai seberapa baik setiap model dalam memprediksi konsumsi energi dan memilih model terbaik yang paling akurat dan andal.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah **"Energy Consumption Prediction"** yang tersedia secara publik di Kaggle. Dataset ini bertujuan untuk membantu memahami pola konsumsi energi berdasarkan kombinasi faktor lingkungan, waktu, dan operasional bangunan. Dataset dapat diakses melalui tautan berikut:  
👉 [Kaggle - Energy Consumption Prediction](https://www.kaggle.com/datasets/ajinilpatel/energy-consumption-prediction/data)

Dataset ini terdiri dari **5000 data observasi** dan **12 kolom fitur**, termasuk satu variabel target (`EnergyConsumption`). Data ini merepresentasikan kombinasi parameter waktu, kondisi lingkungan, dan penggunaan sistem energi di dalam bangunan. Kondisi dataset sangat baik:  
- Tidak terdapat *missing values*  
- Tidak ditemukan data duplikat  
- Tidak terdapat *outliers* yang signifikan

### Variabel dalam Dataset

| Nama Variabel      | Tipe Data   | Deskripsi                                                                 |
|--------------------|-------------|---------------------------------------------------------------------------|
| Month              | Numerik     | Menunjukkan bulan dalam setahun (1–12), berguna untuk analisis musiman.  |
| Hour               | Numerik     | Menunjukkan jam dalam satu hari (0–23), berguna untuk identifikasi pola waktu. |
| DayOfWeek          | Object      | Hari dalam seminggu (Senin–Minggu), dapat memengaruhi pola konsumsi.     |
| Holiday            | Object      | Menunjukkan apakah hari tersebut hari libur atau bukan.                   |
| Temperature        | Numerik     | Suhu lingkungan (°C), mempengaruhi penggunaan HVAC.                       |
| Humidity           | Numerik     | Tingkat kelembapan udara (%), juga berdampak pada HVAC.                   |
| SquareFootage      | Numerik     | Luas bangunan dalam kaki persegi, berpengaruh langsung terhadap energi.  |
| Occupancy          | Numerik     | Jumlah orang di dalam bangunan, memengaruhi sistem energi yang berjalan. |
| HVACUsage          | Object      | Tingkat penggunaan sistem HVAC.                                           |
| LightingUsage      | Object      | Tingkat penggunaan sistem pencahayaan.                                    |
| RenewableEnergy    | Numerik     | Persentase kontribusi energi dari sumber terbarukan.                      |
| EnergyConsumption  | Numerik     | **Target** – Total energi yang dikonsumsi. 

### Visualisasi Data (EDA)
Untuk memperkuat pemahaman terhadap karakteristik dataset, berikut adalah beberapa visualisasi yang dilakukan:
#### 1. Distribusi Fitur Numerik
![Distribusi Numerik](images/gambar%201.png)

Visualisasi ini menunjukkan distribusi dari fitur numerik seperti `Month`, `Hour`, `Temperature`, `Humidity`, `SquareFootage`, `Occupancy`, `RenewableEnergy`, dan `EnergyConsumption`. Sebagian besar fitur memiliki distribusi yang cukup merata, dan target `EnergyConsumption` menunjukkan pola mendekati distribusi normal.

---

#### 2. Distribusi Fitur Kategorikal

![Distribusi Kategorikal](images/gambar%202.png)

Visualisasi ini memperlihatkan persebaran nilai dari fitur kategorikal seperti `DayOfWeek`, `Holiday`, `HVACUsage`, dan `LightingUsage`. Jumlah data cenderung seimbang di setiap kategori, yang baik untuk keperluan pemodelan.

---

#### 3. Rata-rata Konsumsi Energi per Hari

![Rata-rata Konsumsi Energi](images/gambar%203.png)

Grafik ini menunjukkan rata-rata konsumsi energi (`EnergyConsumption`) berdasarkan hari dalam seminggu. Terlihat bahwa variasi antar hari relatif kecil, namun terdapat sedikit peningkatan pada hari Jumat dan Sabtu.

---

#### 4. Visualisasi Hubungan Antar Fitur Numerik

![Pairplot Fitur Numerik](images/gambar%204.png)

Melalui pairplot ini, dapat dilihat bagaimana hubungan antar fitur numerik dan distribusinya. Beberapa fitur memiliki hubungan linier lemah terhadap target, namun distribusi data cukup baik tanpa outliers ekstrem.

---

#### 5. Korelasi Antar Fitur

![Correlation Matrix](images/gambar%205.png)

Heatmap korelasi ini menunjukkan bahwa `Temperature` memiliki korelasi cukup kuat dengan `EnergyConsumption` dibanding fitur lainnya. `Occupancy` dan `RenewableEnergy` juga memiliki korelasi positif yang cukup berarti, sementara fitur waktu seperti `Month` dan `Hour` memiliki korelasi sangat lemah.

## Data Preparation

Pada tahap ini, dilakukan beberapa proses persiapan data sebelum diterapkan pada model machine learning. Langkah-langkah yang dilakukan secara berurutan sebagai berikut:

### 1. **Pengecekan Missing Value**
Langkah pertama adalah memeriksa apakah terdapat nilai yang hilang dalam dataset. Ini penting karena data yang hilang dapat memengaruhi kinerja model prediksi. Hasil pemeriksaan menunjukkan bahwa tidak terdapat missing value di dalam dataset.

### 2. **Encoding Variabel Kategorikal**
Dataset mengandung beberapa fitur kategorikal seperti `DayOfWeek`, `Holiday`, `HVACUsage`, dan `LightingUsage`. Variabel-variabel ini perlu dikonversi ke bentuk numerik agar dapat digunakan dalam algoritma machine learning. Seluruh variabel kategorikal dikodekan menggunakan **Label Encoding**, karena pendekatan ini cukup efektif untuk model-model yang digunakan dalam proyek ini, serta mempertahankan bentuk sederhana dari dataset.

### 3. **Feature Scaling**
Beberapa fitur numerik seperti `Temperature`, `Humidity`, `SquareFootage`, `Occupancy`, dan `RenewableEnergy` memiliki rentang nilai yang berbeda-beda. Untuk menormalkan skala data, digunakan metode **MinMaxScaler** agar semua nilai berada pada rentang 0 hingga 1. Ini dilakukan untuk meningkatkan stabilitas dan konvergensi model, terutama bagi model yang sensitif terhadap skala seperti KNN atau regresi linier.

### 4. **Pemisahan Fitur dan Target**
Fitur input (`X`) terdiri dari seluruh kolom kecuali `EnergyConsumption` yang merupakan target prediksi (`y`).

### 5. **Split Data: Train dan Test**
Dataset dibagi menjadi data latih dan data uji dengan proporsi 80:20 menggunakan `train_test_split` dari scikit-learn. Ini dilakukan agar model dapat dilatih pada sebagian besar data dan diuji performanya pada data yang belum pernah dilihat.

## Modeling

Pada tahap ini, dilakukan pemodelan terhadap data dengan menggunakan enam algoritma regresi untuk memprediksi nilai `EnergyConsumption` sebagai variabel target. Model yang digunakan meliputi:

- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **XGBoost Regressor**

### Tahapan Modeling

Semua model dilatih menggunakan data latih (`x_train`, `y_train`) dan diuji performanya pada data uji (`x_test`, `y_test`). Evaluasi dilakukan menggunakan metrik:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

### Parameter yang Digunakan

- **Linear Regression, Gradient Boosting, Random Forest, XGBoost**: digunakan dengan parameter default.
- **Ridge & Lasso Regression**: digunakan dengan parameter `alpha=1.0`.

### Kelebihan dan Kekurangan Model

| Model | Kelebihan | Kekurangan |
|-------|-----------|------------|
| Linear Regression | Sederhana, cepat, interpretatif | Tidak menangkap non-linearitas |
| Ridge | Mengurangi multikolinearitas | Masih sensitif terhadap outlier |
| Lasso | Melakukan seleksi fitur | Bisa mengeliminasi fitur penting |
| Random Forest | Akurat, tahan terhadap overfitting | Interpretabilitas rendah, lebih lambat |
| Gradient Boosting | Performa tinggi pada banyak kasus | Lebih kompleks dan rentan overfitting jika tidak diatur |
| XGBoost | Cepat dan powerful, cocok untuk dataset besar | Perlu tuning parameter untuk performa optimal |

### Pemilihan Model Terbaik

Dari hasil evaluasi pada notebook, model terbaik dipilih berdasarkan kombinasi nilai **R² tertinggi dan error (MSE, RMSE, MAE) terendah** pada data uji (`test`). Berdasarkan hasil tersebut, **[Gradient Boosting]** dipilih sebagai model akhir karena memberikan performa paling optimal dalam memprediksi konsumsi energi.



## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

