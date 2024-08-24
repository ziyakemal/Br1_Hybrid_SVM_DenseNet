import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# İşlenecek tek bir resmin yolunu belirtin
image_path = "dataSet/YES/TCGA_CS_4941_19960909_14.tif"

# Çıktı klasörünü projenin ana dizini olarak ayarlayın
output_folder = os.getcwd()  # Projenin ana dizinini çıktı klasörü olarak ayarlayın

# Görüntüyü yükleyin
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

# Kırpılmış resmi kaydedin
cropped_image_path = os.path.join(output_folder, "cropped_sample_image.jpg")
cv2.imwrite(cropped_image_path, cropped_image)

# Orijinal ve kırpılmış resimleri doğrulamak için görüntüleyin
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Cropped Image")
plt.imshow(cropped_image, cmap="gray")

# plt.show() yerine kaydetme işlemi
output_plot_path = os.path.join(output_folder, "comparison_plot.png")
plt.savefig(output_plot_path)

# plt.show()
