#
# * 1-) --------------------------------------------------------------------------------------------------------
# Kütüphanelerimizi aşağıdaki gibi import edelim.
# Data Manipulation
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd

# Data preprocessing
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# from PIL import Image, ImageEnhance
from sklearn.utils import shuffle

# For Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data Evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# For ML Models
from keras.applications import Xception, ResNet50, MobileNet, VGG16
import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.losses import *
from keras.models import *
from keras.metrics import *
from keras.optimizers import *
from keras.optimizers import Adam
from keras.applications import *
from keras.preprocessing.image import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical


# Miscellaneous
import cv2

# from tqdm import tqdm
import os
import random
import pickle
from contextlib import redirect_stdout

# & ____________________________________ Tüm çıktılar aşağıdaki path'e kaydedilecek  ________________________________________

# Output directory belirleme
output_directory = "Fine Tunning Models/Models And History/DenseNet121"

# Eğer output directory yoksa oluştur
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# & ____________________________________ DataFrame Oluşturma Fonksiyonu ________________________________________
# * 2-) --------------------------------------------------------------------------------------------------------
# Train, test ve validation dataların path'ini aşağıdaki ilgili değişkenlere atayalım.


def create_dataframe(data_path):
    filepath = []
    label = []
    image_folder = os.listdir(data_path)
    for folder in image_folder:
        folder_path = os.path.join(data_path, folder)
        filelist = os.listdir(folder_path)
        for file in filelist:
            new_path = os.path.join(folder_path, file)
            filepath.append(new_path)
            label.append(folder)
    image_data = pd.Series(filepath, name="image_data")
    label_data = pd.Series(label, name="label")
    df = pd.concat([image_data, label_data], axis=1)
    return df


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# & ____________________________________ DataFramelerin Elde Edilmesi __________________________________________
# * 3-) --------------------------------------------------------------------------------------------------------


train_data = "dataSet_Final/Train"
test_data = "dataSet_Final/Test"
valid_data = "dataSet_Final/Validation"

train_df = create_dataframe(train_data)
test_df = create_dataframe(test_data)
valid_df = create_dataframe(valid_data)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# & ___________________________ Betimsel İstatistik Fonksiyonu ve Elde Edilmesi _________________________________
# * 4-) --------------------------------------------------------------------------------------------------------


# _____________ Train, Test & Validation Seti İçin Betimsel İstatistik _____________
def print_dataset_statistics(df, name):
    print(f"{name} DataFrame:\n", df.head())
    print(f"{name} seti boyutları--> \n", df.shape)
    print(f"Eksik veri gözlemleri--> \n", df.isnull().sum())
    print(f"Kanser Türü Sayıları--> \n", df["label"].value_counts())


print_dataset_statistics(train_df, "Eğitim")
print_dataset_statistics(valid_df, "Eğitim")
print_dataset_statistics(valid_df, "Eğitim")

# * 5-) --------------------------------------------------------------------------------------------------------
# Dataset istatistiklerinin txt doyyası olarak kayıt edilmesi.


def save_statistics_to_file(df, name, save_path):
    # Dosya yolunu oluşturmak için klasör yolunu al
    folder_path = os.path.dirname(save_path)

    # Klasör yoksa oluştur
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Dosyayı oluştur ve betimsel istatistikleri yaz
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"{name} DataFrame:\n")
        f.write(df.head().to_string())
        f.write("\n\n")
        f.write(f"{name} seti boyutları--> \n{df.shape}\n\n")
        f.write(f"Eksik veri gözlemleri--> \n{df.isnull().sum().to_string()}\n\n")
        f.write(
            f"Kanser Türü Sayıları--> \n{df['label'].value_counts().to_string()}\n\n"
        )


# Örnek kullanımı:
output_path = "Fine Tunning Models/Figures And Tables/DenseNet121/validationStatistic.txt"  # Bu yola istediğiniz klasör yolunu yazın
save_statistics_to_file(valid_df, "Validasyon", output_path)

output_path = "Fine Tunning Models/Figures And Tables/DenseNet121/trainStatistic.txt"
save_statistics_to_file(train_df, "Eğitim", output_path)

output_path = "Fine Tunning Models/Figures And Tables/DenseNet121/testStatistic.txt"
save_statistics_to_file(test_df, "Test", output_path)


# ! +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Buradan sonra yer alan yapıda  ImageDataGenerator kullanımı tamamen kaldırılmıştır. Bunun yerine, veriler doğrudan DataFrame'den yüklenip,
# NumPy dizilerine dönüştürülmüş ve bu şekilde modele verilmiştir. Bu yöntemle, augmentation işlemi olmadan, veriler DenseNet modelinde kullanılacaktır.


# & _________________________ Görüntülerin yüklenmesi ve DataFrame'den NumPy dizilerine dönüştürülmesi _______________________________
# * 6-) ------------------------------------------------------------------------------------------------------------------------------
def load_images_from_dataframe(df, target_size=(128, 128)):
    images = []
    labels = []
    for index, row in df.iterrows():
        img_path = row["image_data"]
        label = row["label"]
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, target_size)
            img = img / 255.0  # Normalizasyon
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)


# & _______________________________________ Verilerin yüklenmesi _______________________________________________
# * 7-) --------------------------------------------------------------------------------------------------------

# Verilerin yüklenmesi
train_images, train_labels = load_images_from_dataframe(train_df)
test_images, test_labels = load_images_from_dataframe(test_df)
valid_images, valid_labels = load_images_from_dataframe(valid_df)

# & ___________________________________ Etiketlerin Binarize Edilmesi _____________________________________________
# * 8-) -----------------------------------------------------------------------------------------------------------

# Etiketlerin label_mapping ile sayısal değerlere çevrilmesi
label_mapping = {label: idx for idx, label in enumerate(train_df["label"].unique())}
train_labels = np.array([label_mapping[label] for label in train_labels])
test_labels = np.array([label_mapping[label] for label in test_labels])
valid_labels = np.array([label_mapping[label] for label in valid_labels])

# to_categorical ile one-hot encoding yapılması
train_labels = to_categorical(train_labels, num_classes=len(label_mapping))
test_labels = to_categorical(test_labels, num_classes=len(label_mapping))
valid_labels = to_categorical(valid_labels, num_classes=len(label_mapping))

num_classes = len(label_mapping)

# & _________________________________________ Model Development ________________________________________________
# * 9-) --------------------------------------------------------------------------------------------------------


# DenseNet Modelinin Oluşturulması
def create_densenet169_model(input_shape, num_classes):
    base_model = DenseNet169(
        weights="imagenet", include_top=False, input_shape=input_shape
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    return model


# Modelin oluşturulması
image_shape = (128, 128, 3)
model = create_densenet169_model(image_shape, num_classes)

# Modelin derlenmesi
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# & ______________________________________ Model Özetini Kaydetme ______________________________________________
# * 10-) --------------------------------------------------------------------------------------------------------


summary_file_path = os.path.join(output_directory, "model_summary.txt")
with open(summary_file_path, "w") as f:
    with redirect_stdout(f):
        model.summary()
print(f"Model summary saved to --> {summary_file_path}")

# & _________________________________ MODELİ EĞİTME & CALLBACK _________________________________________
# * 11-) ------------------------------------------------------------------------------------------------

ckp_interval = 5 * int(np.ceil(train_df.shape[0] / 64))
ckp_path = os.path.join(
    output_directory, r"epocch_{epoch:02d}_val_acc_{val_accuracy:.2f}.hdf5"
)

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=ckp_path,
    save_weights_only=True,
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6),
    checkpoint,
]

history = model.fit(
    train_images,
    train_labels,
    validation_data=(valid_images, valid_labels),
    epochs=1,
    batch_size=64,
    callbacks=callbacks,
)

# & ___________________________ History Dosyasının Kayıt Edilmesi ______________________________________
# * 12-) ------------------------------------------------------------------------------------------------

# History dosyasının kaydedilmesi
history_file_path = os.path.join(output_directory, "history_MS.npy")
with open(history_file_path, "wb") as f:
    np.save(f, history.history)

print(f"History saved to --> {history_file_path}")

# & _________________________ Eğitim Bitiminde Modelin Kayıt Edilmesi __________________________________
# * 13-) ------------------------------------------------------------------------------------------------

# Modelin kaydedilmesi
model_save_path = os.path.join(output_directory, "MS_classifier_MS.keras")
model.save(model_save_path)
print(f"Model saved to --> {model_save_path}")

# Modelin test edilmesi
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc}")
