import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Girdi klasörünü belirtin
input_folder = "dataSet\YES"
# Çıktı klasörünü projenin ana dizini olarak ayarlayın
output_folder = os.path.join(os.getcwd(), "dataSet_Cropped\YES_Cropped")

# Çıktı klasörünün var olup olmadığını kontrol edin, yoksa oluşturun
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Klasördeki tüm .tif uzantılı resimleri işleyin
for filename in os.listdir(input_folder):
    if filename.endswith(".tif"):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Kenarları tespit etmek için Canny kenar algılayıcıyı kullanın
        edges = cv2.Canny(image, 50, 150)

        # Morfolojik işlemler için kernel oluşturun
        kernel = np.ones((5, 5), np.uint8)

        # Dilation ve erosion işlemleri ile kenarları genişletin ve kapatın
        dilated = cv2.dilate(edges, kernel, iterations=2)
        eroded = cv2.erode(dilated, kernel, iterations=1)

        # Konturları bulun
        contours, _ = cv2.findContours(
            eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Konturların boş olup olmadığını kontrol edin
        if contours:
            # En büyük konturu bulun (bu kontur beyne karşılık gelmelidir)
            largest_contour = max(contours, key=cv2.contourArea)

            # En büyük kontur etrafında sınırlayıcı kutunun koordinatlarını alın
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Orijinal gri tonlamalı görüntüyü sınırlayıcı kutuya göre kırpın
            cropped_image = image[y : y + h, x : x + w]

            # Kırpılmış resmi kaydedin
            cropped_image_path = os.path.join(output_folder, f"cropped_{filename}")
            cv2.imwrite(cropped_image_path, cropped_image)
        else:
            print(f"Kontur bulunamadı: {filename}")


# import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plt

# # Girdi klasörünü belirtin
# input_folder = "C:\Users\ziyak\Documents\GitHub\Br1_Hybrid_SVM_DenseNet\dataSet\NO"
# # Çıktı klasörünü projenin ana dizini olarak ayarlayın
# output_folder = os.path.join(os.getcwd(), "C:\Users\ziyak\Documents\GitHub\Br1_Hybrid_SVM_DenseNet\dataSet_Cropped\NO_Cropped")

# # Çıktı klasörünün var olup olmadığını kontrol edin, yoksa oluşturun
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # Klasördeki tüm resimleri işleyin
# for filename in os.listdir(input_folder):
#     if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".tif"):
#         image_path = os.path.join(input_folder, filename)
#         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#         # Kenarları tespit etmek için Canny kenar algılayıcıyı kullanın
#         edges = cv2.Canny(image, 50, 150)

#         # Morfolojik işlemler için kernel oluşturun
#         kernel = np.ones((5, 5), np.uint8)

#         # Dilation ve erosion işlemleri ile kenarları genişletin ve kapatın
#         dilated = cv2.dilate(edges, kernel, iterations=2)
#         eroded = cv2.erode(dilated, kernel, iterations=1)

#         # Konturları bulun
#         contours, _ = cv2.findContours(
#             eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#         )

#         # En büyük konturu bulun (bu kontur beyne karşılık gelmelidir)
#         largest_contour = max(contours, key=cv2.contourArea)

#         # En büyük kontur etrafında sınırlayıcı kutunun koordinatlarını alın
#         x, y, w, h = cv2.boundingRect(largest_contour)

#         # Orijinal gri tonlamalı görüntüyü sınırlayıcı kutuya göre kırpın
#         cropped_image = image[y : y + h, x : x + w]

#         # Kırpılmış resmi kaydedin
#         cropped_image_path = os.path.join(output_folder, f"cropped_{filename}")
#         cv2.imwrite(cropped_image_path, cropped_image)
