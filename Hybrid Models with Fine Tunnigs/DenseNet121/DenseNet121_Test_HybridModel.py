# * 1-) --------------------------------------------------------------------------------------------------------
# Kütüphanelerimizi aşağıdaki gibi import edelim.
# Data Manipulation

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import pandas as pd

# Data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle

# For Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data Evaluation
from sklearn.metrics import confusion_matrix, classification_report

# For ML Models
from keras.applications import DenseNet121
import tensorflow as tf
from tensorflow import keras
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical

# Miscellaneous
import cv2
import os
import random
import pickle
from contextlib import redirect_stdout

# & ____________________________________ Tüm çıktılar aşağıdaki path'e kaydedilecek  ________________________________________

# Output directory belirleme
output_directory = "Hybrid Models with Fine Tunnigs/DenseNet121/Models And History"

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


# & ____________________________________ DataFramelerin Elde Edilmesi __________________________________________
# * 3-) --------------------------------------------------------------------------------------------------------

train_data = "dataSet_Final/Train"
test_data = "dataSet_Final/Test"
valid_data = "dataSet_Final/Validation"

train_df = create_dataframe(train_data)
test_df = create_dataframe(test_data)
valid_df = create_dataframe(valid_data)

# & ___________________________ Betimsel İstatistik Fonksiyonu ve Elde Edilmesi _________________________________
# * 4-) --------------------------------------------------------------------------------------------------------


# _____________ Train, Test & Validation Seti İçin Betimsel İstatistik _____________
def print_dataset_statistics(df, name):
    print(f"{name} DataFrame:\n", df.head())
    print(f"{name} seti boyutları--> \n", df.shape)
    print(f"Eksik veri gözlemleri--> \n", df.isnull().sum())
    print(f"Kanser Türü Sayıları--> \n", df["label"].value_counts())


print_dataset_statistics(train_df, "Eğitim")
print_dataset_statistics(valid_df, "Validasyon")
print_dataset_statistics(test_df, "Test")

# * 5-) --------------------------------------------------------------------------------------------------------
# Dataset istatistiklerinin txt dosyası olarak kayıt edilmesi.


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
# Bu yola istediğiniz klasör yolunu yazın
output_path = "Hybrid Models with Fine Tunnigs/DenseNet121/Figure And Tables/validationStatistic.txt"
save_statistics_to_file(valid_df, "Validasyon", output_path)

output_path = (
    "Hybrid Models with Fine Tunnigs/DenseNet121/Figure And Tables/trainStatistic.txt"
)
save_statistics_to_file(train_df, "Eğitim", output_path)

output_path = (
    "Hybrid Models with Fine Tunnigs/DenseNet121/Figure And Tables/testStatistic.txt"
)
save_statistics_to_file(test_df, "Test", output_path)


# ! +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Buradan sonra yer alan yapıda  ImageDataGenerator kullanımı tamamen kaldırılmıştır. Bunun yerine, veriler doğrudan DataFrame'den yüklenip,
# NumPy dizilerine dönüştürülmüş ve bu şekilde modele verilmiştir. Bu yöntemle, augmentation işlemi olmadan, veriler DenseNet modelinde kullanılacaktır.


# & _________________________ Görüntülerin yüklenmesi ve DataFrame'den NumPy dizilerine dönüştürülmesi _______________________________
# * 6-) ------------------------------------------------------------------------------------------------------------------------------
def load_images_from_dataframe(df, target_size=(224, 224)):
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
def create_densenet121_model(input_shape, num_classes):
    # ^ 9.1. Pre-trained Modelin Yüklenmesi ve Son Katmanlarının Dondurulması
    base_model = DenseNet121(
        weights="imagenet", include_top=False, input_shape=input_shape
    )

    # ^ 9.2. Son Katmanların Eklenmesi
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)  # overfittingin önlenmesi için
    predictions = Dense(num_classes, activation="softmax")(x)

    # ^ 9.3. Modelin Tanımlanması
    model = Model(inputs=base_model.input, outputs=predictions)

    # ^ 9.4. Önceden Eğitilmiş Katmanların Dondurulması
    for layer in base_model.layers:
        layer.trainable = False

    return model


# Modelin oluşturulması
image_shape = (224, 224, 3)  # DenseNet121 için gerekli boyut
model = create_densenet121_model(image_shape, num_classes)

# ^ 9.5. Modelin derlenmesi
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# & ______________________________________ Model Özetini Kaydetme ______________________________________________
# * 10-) --------------------------------------------------------------------------------------------------------

summary_file_path = os.path.join(
    output_directory, "DenseNet121_SVM_HybridModel_summary.txt"
)
with open(summary_file_path, "w") as f:
    with redirect_stdout(f):
        model.summary()
print(f"Model summary saved to --> {summary_file_path}")

# & _________________________________ MODELİ EĞİTME & CALLBACK _________________________________________
# * 11-) ------------------------------------------------------------------------------------------------


def create_callbacks(
    output_directory,
    train_df_shape,
    checkpoint_name="epoch_{epoch:02d}_val_acc_{val_accuracy:.2f}.hdf5",
    patience=5,
    min_lr=1e-6,
):
    """
    Model eğitiminde kullanılacak callback'leri oluşturur ve checkpoint dosya adını özelleştirir.

    Parameters:
    - output_directory (str): Checkpoint dosyalarının kaydedileceği dizin.
    - train_df_shape (int): Eğitim veri kümesinin boyutu.
    - checkpoint_name (str): Checkpoint dosyası için kullanılacak isim şablonu.
    - patience (int): EarlyStopping için sabır süresi.
    - min_lr (float): ReduceLROnPlateau için minimum öğrenme oranı.

    Returns:
    - callbacks (list): Model eğitiminde kullanılacak callback'ler listesi.
    """

    ckp_interval = 5 * int(np.ceil(train_df_shape / 64))
    ckp_path = os.path.join(output_directory, checkpoint_name)

    checkpoint = ModelCheckpoint(
        filepath=ckp_path,
        save_weights_only=True,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=patience, min_lr=min_lr
        ),
        checkpoint,
    ]

    return callbacks


# ~ +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

callbacks = create_callbacks(
    output_directory,
    train_df.shape[0],
    checkpoint_name="Before_FineTunning_checkpoint_epoch_{epoch:02d}_accuracy_{val_accuracy:.2f}.hdf5"
)

# ^ 9.6. Fine-Tuning Öncesi Eğitim (Son Katmanların Eğitimi)

history = model.fit(
    train_images,
    train_labels,
    validation_data=(valid_images, valid_labels),
    epochs=5,
    batch_size=64,
    callbacks=callbacks,
)

# & ___________________________ History Dosyasının Kayıt Edilmesi ______________________________________
# * 12-) ------------------------------------------------------------------------------------------------

# History dosyasının kaydedilmesi
history_file_path = os.path.join(
    output_directory, "history_DenseNet121_sonKatmanlarinEgitimi.npy"
)
with open(history_file_path, "wb") as f:
    np.save(f, history.history)

print(f"History saved to --> {history_file_path}")

# ^ 9.7. Fine-Tuning İşlemi
# Üst katmanların da eğitilebilir hale getirilmesi (örneğin, son birkaç katman)
for layer in model.layers[-30:]:  # Son 30 katman eğitilebilir hale getirildi
    layer.trainable = True

# ^ 9.8. Modelin Yeniden Derlenmesi (Öğrenme oranı daha düşük olabilir)
model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# ^ 9.9. Fine-Tuning Eğitimi
callbacks = create_callbacks(
    output_directory,
    train_df.shape[0],
    checkpoint_name="After_FineTunning_checkpoint_epoch_{epoch:02d}_accuracy_{val_accuracy:.2f}.hdf5"
)

history = model.fit(
    train_images,
    train_labels,
    validation_data=(valid_images, valid_labels),
    epochs=5,
    batch_size=64,
    callbacks=callbacks,
)

# History dosyasının kaydedilmesi
history_file_path = os.path.join(
    output_directory, "history_DenseNet121_fineTunningEgitimi.npy"
)
with open(history_file_path, "wb") as f:
    np.save(f, history.history)

print(f"History saved to --> {history_file_path}")

# & _________________________ Eğitim Bitiminde Modelin Kayıt Edilmesi __________________________________
# * 13-) ------------------------------------------------------------------------------------------------

# Modelin kaydedilmesi
model_save_path = os.path.join(output_directory, "DenseNet121_classifier.keras")
model.save(model_save_path)
print(f"Model saved to --> {model_save_path}")

# Modelin test edilmesi
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc}")

# & ___________________________ NumPy Dizilerinin Betimsel İstatistiklerinin Kayıt Edilmesi _________________________________
# * 14-) --------------------------------------------------------------------------------------------------------


def save_numpy_statistics(images, labels, save_path, dataset_name):
    # Dosya yolunu oluşturmak için klasör yolunu al
    folder_path = os.path.dirname(save_path)

    # Klasör yoksa oluştur
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Dosyayı oluştur ve betimsel istatistikleri yaz
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"{dataset_name} Dataset Betimsel İstatistikler:\n")
        f.write(f"Images shape: {images.shape}\n")
        f.write(f"Labels shape: {labels.shape}\n")
        f.write(f"Unique Labels: {np.unique(np.argmax(labels, axis=1))}\n")
        f.write(f"Labels distribution:\n{np.sum(labels, axis=0)}\n")


# & ___________________________ Tüm Datasetlerin Betimsel İstatistiklerini Kaydet ___________________________________________

# Train seti için
train_images_stats_path = "Hybrid Models with Fine Tunnigs/DenseNet121/Figure And Tables/train_images_statistics.txt"
save_numpy_statistics(train_images, train_labels, train_images_stats_path, "Train")

# Validation seti için
valid_images_stats_path = "Hybrid Models with Fine Tunnigs/DenseNet121/Figure And Tables/valid_images_statistics.txt"
save_numpy_statistics(valid_images, valid_labels, valid_images_stats_path, "Validation")

# Test seti için
test_images_stats_path = "Hybrid Models with Fine Tunnigs/DenseNet121/Figure And Tables/test_images_statistics.txt"
save_numpy_statistics(test_images, test_labels, test_images_stats_path, "Test")
