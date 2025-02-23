# İTÜ AI Datathon – Görselden Konum Tahmini

Bu repo, İTÜ (İstanbul Teknik Üniversitesi) kampüsünde çekilen fotoğraflardan **konum** (enlem-boylam) tahmini yapmayı amaçlayan bir projeyi içerir. Proje kapsamında hem **derin öğrenme** tabanlı **özellik çıkarma** hem de **kümeleme** ve **sınıflandırma** yöntemleriyle nihai konum tahmini yapılmaktadır. Kod, Datathon yarışmasında **8.** olan ekibin çözüm yaklaşımını göstermektedir.

---

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Veri ve Ön İşleme](#veri-ve-ön-i̇şleme)
- [Model ve Yaklaşım](#model-ve-yaklaşım)
  - [ResNet50 Özellik Çıkarma + Spektral Kümeleme](#resnet50-özellik-çıkarma--spektral-kümeleme)
  - [ResNet18 Sınıflandırma ve Embedding Çıkarma](#resnet18-sınıflandırma-ve-embedding-çıkarma)
  - [Konum İyileştirme (Refinement)](#konum-iyileştirme-refinement)
- [Gereksinimler](#gereksinimler)
- [Proje Yapısı](#proje-yapısı)
- [Sonuçlar](#sonuçlar)

---

## Genel Bakış

Yarışma kapsamında İTÜ kampüsünde belirli noktalarda çekilmiş fotoğrafların **görsel** (piksel bilgisi) + **enlem** ve **boylam** bilgileri verilmişti. Amacımız, **yeni** bir fotoğraf için **konum** (enlem-boylam) tahmini yapmaktı.

Bu proje:
1. **İlk aşamada** (ilk kod bloğu) ResNet50 ile özellik (feature) çıkarma ve ardından Spektral Kümeleme (Spectral Clustering) kullanarak fotoğrafları alt kümelere ayırıyor.
2. **İkinci aşamada** (`train_pipeline.py`) bir sınıflandırma modeli (ResNet18) eğiterek, görüntüden hangi cluster’a (küme) ait olduğu tahmin ediliyor. Bu sayede daha **ince** konum tahmini yapılabilmesi hedefleniyor.
3. **Son olarak** ek bir adım ile tahmin edilen cluster içindeki en yakın embedding (özellik vektörü) bulunarak koordinatlar daha da hassaslaştırılıyor (Refine).

---

## Veri ve Ön İşleme

- **Veri Kaynağı**: İTÜ kampüsünde çekilmiş fotoğraflar. Her fotoğrafla birlikte `filename, latitude, longitude` alanları bulunuyor.
- **Ön İşleme**:
  - Görseller, `PIL` ile yüklenip `torchvision.transforms` ile yeniden boyutlandırılıyor (512x512) ve normalize ediliyor.
  - Latitude/Longitude değerleri Haversine mesafesi için radyana dönüştürülüyor.
  - Bazı kod bloklarında veri `train` ve `test` olarak ayrılmıştır.

---

## Model ve Yaklaşım

### ResNet50 Özellik Çıkarma + Spektral Kümeleme

İlk kod bloğu şu adımları içerir:

1. **ResNet50’yi yükleyerek** son katmanını (fc) kaldırıyoruz. Böylece 2048 boyutunda bir özellik vektörü elde edebiliyoruz.
2. **L2 mesafesi** (Görüntü özellikleri arasında) ve **Haversine mesafesi** (Coğrafi konumlar arasında) hesaplanıyor.
3. Mesafeler normalize edilip **kombine** edilerek **Spektral Kümeleme** uygulanıyor.
4. Nihai olarak, her fotoğrafın ait olduğu küme belirleniyor. Bu küme indeksleri, konumun tahmini için bir başlangıç noktası veriyor.

### ResNet18 Sınıflandırma ve Embedding Çıkarma

`train_pipeline.py` içindeki ana aşamalar:

1. **Dataset Sınıfı (GeoClusterDataset)**:  
   - Veriyi (`filename, latitude, longitude, cluster`) okur.  
   - Görseli `PIL` ile yükler ve istenen dönüşümleri uygular.  
   - Hedef label olarak `cluster` verisini kullanır.

2. **Model Oluşturma (create_model)**:  
   - `torchvision.models.resnet18(pretrained=True)` yüklenir.  
   - Son katman (fc) `num_clusters` kadar çıktıya sahip olacak şekilde ayarlanır.

3. **Eğitim**:  
   - `train_one_epoch` fonksiyonuyla eğitim döngüsü,  
   - `evaluate` fonksiyonuyla validasyon döngüsü yapılır.  
   - Kayıp fonksiyonu olarak `CrossEntropyLoss`, optimizasyon olarak `Adam` tercih edilir.

4. **Embedding Çıkarma (extract_cluster_embeddings)**:  
   - Eğitilen modelin son katmanı **hariç** tutularak, her görsel için 512 boyutlu bir **embedding** elde edilir.  
   - Cluster bazında bu embedding’ler saklanır.

### Konum İyileştirme (Refinement)

- **Refine aşamasında** yeni bir test görüntüsü için önce hangi cluster’a ait olduğu tahmin edilir.  
- Ardından o cluster’daki tüm embedding’ler ile L2 mesafesi karşılaştırılarak **en yakın** embedding bulunur.  
- En yakın embedding’in `latitude, longitude` bilgisi, tahmin olarak atanır.

Bu sayede sadece sınıflandırma çıktısı (cluster) değil, o cluster içindeki **en yakın** noktanın koordinatı sonuç olarak döndürülür.

---

## Gereksinimler

Aşağıdaki kütüphanelerin kurulmuş olması gerekir (minimum sürümler tahmini olarak verilmiştir):

```bash
Python 3.7+
numpy >= 1.19.5
pandas >= 1.1.5
scikit-learn >= 0.24.2
torch >= 1.9.0
torchvision >= 0.10.0
PIL (Pillow) >= 8.0.0
matplotlib >= 3.2.2
```

Ek olarak, GPU kullanımı için **CUDA** destekli bir ortam önerilir.

---

## Proje Yapısı

Aşağıdaki yapı örnektir. Dosyalar ve klasör isimleri değişiklik gösterebilir:

```
itu-ai-datathon/
├── train.csv
├── test.csv
├── train/                 # Eğitim görselleri
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── test/                  # Test görselleri
│   ├── imgA.jpg
│   ├── imgB.jpg
│   └── ...
├── feature_clustering.py  # (İlk paylaşılan kod) ResNet50 + Spektral Kümeleme
├── train_pipeline.py      # (İkinci kod) ResNet18 + cluster classification
├── requirements.txt       # Gerekli paketler
└── README.md
```

---

## Sonuçlar

- Proje, Datathon yarışmasında **8.** sıraya yerleşti (30+ katılımcı arasından).  
- **Haversine RMSE** gibi metriklerle konum hataları ölçüldü.  
- Son aşamada tahminler, cluster bazında embedding benzerliğiyle iyileştirildi (Refinement).

**Önemli Noktalar**  
- Embedding tabanlı yaklaşımın, doğrudan sınıflandırma + konum atamaya göre daha iyi sonuç verebildiği gözlemlendi.  
- Mesafe matrislerinin normalizasyonu ve ağırlıklı birleşimi (örn. `0.4 * L2 + 0.6 * (1 - Haversine)`) başarıyı etkiledi.  
- Görsellerde veri çeşitliliğinin az olması durumunda overfitting görülme riski var.

---


*Bu README, İTÜ AI Datathon’da kullanılmak üzere hazırlanmış ve yarışma sonrasında referans niteliğinde düzenlenmiştir.*
