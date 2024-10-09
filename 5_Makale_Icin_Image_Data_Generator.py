import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img

# Veri artırma parametrelerini belirleyin
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)


# Görüntüleri kare şeklinde yerleştirmek için işlev
def plot_augmented_images(original_img, augmented_images, titles):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # 2x2 artırılmış görüntüler için

    # Orijinal görüntü alt resimlerin ortasına ek bir ax açarak ortalayacağız
    original_fig = plt.figure(figsize=(10, 10))
    original_ax = original_fig.add_subplot(111)
    original_ax.imshow(original_img)
    original_ax.set_title("Original Image")
    original_ax.axis("off")

    # Artırılmış görüntüleri kare şeklinde yerleştir
    for i, (img, title) in enumerate(zip(augmented_images, titles)):
        row, col = divmod(i, 2)  # 2x2'lik grid için satır ve sütun hesapla
        axes[row, col].imshow(img)
        axes[row, col].set_title(title)
        axes[row, col].axis("off")

    # Orijinal görüntüyü kare ortasına alacak şekilde ayarlayalım
    plt.subplots_adjust(hspace=0.5)
    plt.show()


# Görüntülerin bulunduğu klasörü ve görüntü yolunu belirtin
input_img_path = "dataSet/YES/TCGA_CS_4943_20000902_15.tif"  # Örnek görüntü dosyası

# Görüntüyü yükleyin
img = load_img(input_img_path)
x = img_to_array(img)  # Görüntüyü numpy array'e çevirin
x = x.reshape((1,) + x.shape)  # 4 boyutlu hale getirin

# Orijinal görüntüyü kaydetmek için
original_img = np.array(img, dtype=np.uint8)

# Veri artırma işlemiyle elde edilen görüntüleri depolayın
augmented_images = []
titles = []  # Başlıkları depolamak için
i = 0
for batch in datagen.flow(x, batch_size=1, save_prefix="aug", save_format="jpeg"):
    augmented_images.append(
        np.squeeze(batch, axis=0).astype(np.uint8)
    )  # Boyutu küçült ve numpy array'e dönüştür

    # Artırma işlemlerine göre başlıklar ekleyin
    if i == 0:
        titles.append("Rotation")
    elif i == 1:
        titles.append("Width Shift")
    elif i == 2:
        titles.append("Height Shift")
    elif i == 3:
        titles.append("Horizontal Flip")

    i += 1
    if i >= 4:  # Yalnızca 4 artırılmış görüntü üret
        break

# Orijinal ve artırılmış görüntüleri kare şeklinde göster
plot_augmented_images(original_img, augmented_images, titles)


# ~ Görüntüler yan yana, stabil çalışan kod.
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from keras.preprocessing.image import ImageDataGenerator
# from keras.utils import img_to_array, load_img

# # Veri artırma parametrelerini belirleyin
# datagen = ImageDataGenerator(
#     rotation_range=30,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode="nearest",
# )


# # Görüntüleri yan yana göstermek için işlev
# def plot_augmented_images(original_img, augmented_images, titles):
#     num_images = len(augmented_images) + 1  # Orijinal görüntü dahil
#     fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
#     axes[0].imshow(original_img)
#     axes[0].set_title("Original Image")
#     axes[0].axis("off")

#     for i, (img, title) in enumerate(zip(augmented_images, titles)):
#         axes[i + 1].imshow(img)
#         axes[i + 1].set_title(title)
#         axes[i + 1].axis("off")

#     plt.tight_layout()
#     plt.show()


# # Görüntülerin bulunduğu klasörü ve görüntü yolunu belirtin
# input_img_path = "dataSet\YES\TCGA_CS_4944_20010208_10.tif"  # Örnek görüntü dosyası

# # Görüntüyü yükleyin
# img = load_img(input_img_path)
# x = img_to_array(img)  # Görüntüyü numpy array'e çevirin
# x = x.reshape((1,) + x.shape)  # 4 boyutlu hale getirin

# # Orijinal görüntüyü kaydetmek için
# original_img = np.array(img, dtype=np.uint8)

# # Veri artırma işlemiyle elde edilen görüntüleri depolayın
# augmented_images = []
# titles = []  # Başlıkları depolamak için
# i = 0
# for batch in datagen.flow(x, batch_size=1, save_prefix="aug", save_format="jpeg"):
#     augmented_images.append(
#         np.squeeze(batch, axis=0).astype(np.uint8)
#     )  # Boyutu küçült ve numpy array'e dönüştür

#     # Artırma işlemlerine göre başlıklar ekleyin
#     if i == 0:
#         titles.append("Rotation")
#     elif i == 1:
#         titles.append("Width Shift")
#     elif i == 2:
#         titles.append("Height Shift")
#     elif i == 3:
#         titles.append("Horizontal Flip")

#     i += 1
#     if i >= 4:  # Yalnızca 4 artırılmış görüntü üret
#         break

# # Orijinal ve artırılmış görüntüleri yan yana göster
# plot_augmented_images(original_img, augmented_images, titles)
