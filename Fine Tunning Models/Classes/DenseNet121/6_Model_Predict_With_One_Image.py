import cv2
import numpy as np
import os
from keras.models import load_model


# Kırpma işlemi için gerekli fonksiyon
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Kenarları tespit etmek için Canny kenar algılayıcıyı kullanın
    edges = cv2.Canny(image, 50, 150)

    # Morfolojik işlemler için kernel oluşturun
    kernel = np.ones((5, 5), np.uint8)

    # Dilation ve erosion işlemleri ile kenarları genişletin ve kapatın
    dilated = cv2.dilate(edges, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Konturları bulun
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # En büyük konturu bulun (bu kontur beyne karşılık gelmelidir)
    largest_contour = max(contours, key=cv2.contourArea)

    # En büyük kontur etrafında sınırlayıcı kutunun koordinatlarını alın
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Orijinal gri tonlamalı görüntüyü sınırlayıcı kutuya göre kırpın
    cropped_image = image[y : y + h, x : x + w]

    # Gri tonlamalı görüntüyü üç kanallı hale getirin
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB)

    # Kırpılmış ve üç kanallı hale getirilmiş resmi yeniden boyutlandırın
    resized_image = cv2.resize(cropped_image, (224, 224))
    normalized_image = resized_image / 255.0  # Normalizasyon

    return normalized_image


# Modeli yükleyin
model_path = (
    "Fine Tunning Models/Models And History/DenseNet121/DenseNet121_classifier.keras"
)
model = load_model(model_path)

# label_mapping tanımı (bu tanımı eğitim sırasında nasıl yaptıysanız öyle tanımlayın)
label_mapping = {
    "No": 0,
    "Yes": 1,
}  # Örnek bir label_mapping, sizin kullandığınız mapping ile aynı olmalı

# Görüntü yolunu belirtin
image_path = (
    "your_image_path.tif"  # Buraya tahmin yapmak istediğiniz görüntünün yolunu yazın
)

# Görüntüyü ön işleyin
preprocessed_image = preprocess_image(image_path)

# Modelin beklediği formatta veriyi hazırlayın
input_image = np.expand_dims(preprocessed_image, axis=0)  # 4 boyutlu hale getirin

# Tahmin yapın
predictions = model.predict(input_image)
predicted_class = np.argmax(predictions, axis=1)

# Etiketleri geri döndürmek için label_mapping'i tersine çevirin
inverse_label_mapping = {v: k for k, v in label_mapping.items()}
predicted_label = inverse_label_mapping[predicted_class[0]]

print(f"Görüntünün tahmin edilen sınıfı: {predicted_label}")
