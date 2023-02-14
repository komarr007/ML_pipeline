# Diabetes Detection ML Pipeline

Username dicoding: komarr007

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| Masalah | Diabetes adalah penyakit kronis yang mempengaruhi cara tubuh memproses gula darah, dan itu adalah penyebab utama masalah kesehatan serius, seperti penyakit jantung, kebutaan, gagal ginjal, dan amputasi. Suku Pima Indian memiliki salah satu tingkat diabetes tertinggi di dunia, dan penyakit itu menjadi masalah kesehatan utama bagi suku tersebut.Karakteristik medis yang disertakan dalam dataset termasuk variabel seperti usia, jumlah kehamilan, tekanan darah, indeks massa tubuh, tingkat insulin, ketebalan kulit, dan lain-lain. Variabel-variabel ini dipilih berdasarkan pengetahuan sebelumnya tentang faktor risiko terkait diabetes.  |
| Solusi machine learning | Pada proyek ini membuat model _machine learning_ yang dapat memprediksi apakah seseorang wanita berpotensi menderita penyakit diabetes berdasarkan tingkat glukosa, tekanan darah, insulin, BMI, dan total kehamilan |
| Metode pengolahan | _Dataset_ ini berisikan data dengan tipe numerik, jadi pada proyek ini dilakukan _scaling_ pada fitur dengan menggunakan _z score_|
| Arsitektur model | Model memiliki 8 input layer dan 1 hidden layer dengan aktivasi relu dan 1 layer output dengan aktivasi sigmoid, model sigmoid dipilih untuk output dikarenakan tipe klasifikasi pada data merupakan tipe biner |
| Metrik evaluasi | Pada proyek ini metrik evaluasi _AUC_ dan _Binary Accuracy_ |
| Performa model | Hasil _AUC_ dari model yang telah dibuat memiliki skor 0.821 dan hasil _Binary Accuracy_ sebesar 0.757 |
| Opsi deployment | Pada proyek _deploy_ model dilakukan dengan _free hosting_ dari railway. Railway merupakan platform infrastruktur di mana Anda dapat menyediakan infrastruktur, mengembangkan dengan infrastruktur tersebut secara lokal, dan kemudian melakukan deploy ke cloud. |
| Web app | [Diabetes Detection Model](https://mlpipeline-production.up.railway.app/v1/models/cc-model/metadata)|
| Monitoring | Proyek ini menggunakan prometheus sebagai _tool_ untuk memonitoring model, pada penggunaan _monitoring_ dengan prometheus diapatkan hasil bahwa setiap request dilakukan akan tercatat jumlah request dan jenis metadata mengenai request yang dilakukan. _monitoring_ pada proyek menghasilkan 14 request dengan status OK pada model _machine learning_ yang telah di _deploy_ |
