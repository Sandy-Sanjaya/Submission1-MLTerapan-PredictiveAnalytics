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
![Distribusi Numerik](https://drive.google.com/file/d/1wolGTFC7p-T_zX-KbNNDNdZ8EVep6PkO/view?usp=drive_link)

Visualisasi ini menunjukkan distribusi dari fitur numerik seperti `Month`, `Hour`, `Temperature`, `Humidity`, `SquareFootage`, `Occupancy`, `RenewableEnergy`, dan `EnergyConsumption`. Sebagian besar fitur memiliki distribusi yang cukup merata, dan target `EnergyConsumption` menunjukkan pola mendekati distribusi normal.

---

#### 2. Distribusi Fitur Kategorikal

![Distribusi Kategorikal](29dbebcc-44b1-4323-8d2f-7fde093b9356.png)

Visualisasi ini memperlihatkan persebaran nilai dari fitur kategorikal seperti `DayOfWeek`, `Holiday`, `HVACUsage`, dan `LightingUsage`. Jumlah data cenderung seimbang di setiap kategori, yang baik untuk keperluan pemodelan.

---

#### 3. Rata-rata Konsumsi Energi per Hari

![Rata-rata Konsumsi Energi](5de6012e-a870-4768-b2d0-7c389d1d2a50.png)

Grafik ini menunjukkan rata-rata konsumsi energi (`EnergyConsumption`) berdasarkan hari dalam seminggu. Terlihat bahwa variasi antar hari relatif kecil, namun terdapat sedikit peningkatan pada hari Jumat dan Sabtu.

---

#### 4. Visualisasi Hubungan Antar Fitur Numerik

![Pairplot Fitur Numerik](e619e93b-6de2-435a-976e-e4f84e2fc63e.png)

Melalui pairplot ini, dapat dilihat bagaimana hubungan antar fitur numerik dan distribusinya. Beberapa fitur memiliki hubungan linier lemah terhadap target, namun distribusi data cukup baik tanpa outliers ekstrem.

---

#### 5. Korelasi Antar Fitur

![Correlation Matrix](60c8de9d-9492-4639-838a-e73c3cebc1b7.png)

Heatmap korelasi ini menunjukkan bahwa `Temperature` memiliki korelasi cukup kuat dengan `EnergyConsumption` dibanding fitur lainnya. `Occupancy` dan `RenewableEnergy` juga memiliki korelasi positif yang cukup berarti, sementara fitur waktu seperti `Month` dan `Hour` memiliki korelasi sangat lemah.


## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

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

