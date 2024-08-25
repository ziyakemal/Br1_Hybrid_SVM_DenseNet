import numpy as np  # History dosyasını yüklemek için
import matplotlib.pyplot as plt  # Accuracy, Loss ve ROC grafikleri oluşturmak için
from sklearn.metrics import roc_curve, auc  # ROC eğrisi ve AUC hesaplamaları için
import os  # Dosya yollarını oluşturmak ve dosyaları kaydetmek için

#
# * 1-) --------------------------------------------------------------------------------------------------------
# Kütüphanelerimizi aşağıdaki gibi import edelim.
# Data Manipulation
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd

# Data preprocessing
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# from PIL import Image, ImageEnhance
from sklearn.utils import shuffle

# For Data Visualization
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
import random
import pickle
from contextlib import redirect_stdout

# & ____________________________________ Tüm çıktılar aşağıdaki path'e kaydedilecek  ________________________________________

history_file_path = (
    "Fine Tunning Models/Models And History/DenseNet121/history_DenseNet121.npy"
)
output_directory = "Fine Tunning Models/Figures And Tables/DenseNet121"

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

# & _________________________________________ Model Loading ________________________________________________
# * 9-) --------------------------------------------------------------------------------------------------------


model = load_model(
    "Fine Tunning Models/Models And History/DenseNet121/DenseNet121_classifier.keras"
)

# ! +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# History dosyasının yüklenmesi
loaded_history = np.load(history_file_path, allow_pickle=True).item()

# Accuracy grafiği
plt.figure(figsize=(10, 6))
plt.plot(loaded_history["accuracy"], label="Training Accuracy")
plt.plot(loaded_history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(loc="upper left")
accuracy_plot_path = os.path.join(output_directory, "accuracy_plot.png")
plt.savefig(accuracy_plot_path)
plt.show()

# Loss grafiği
plt.figure(figsize=(10, 6))
plt.plot(loaded_history["loss"], label="Training Loss")
plt.plot(loaded_history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(loc="upper left")
loss_plot_path = os.path.join(output_directory, "loss_plot.png")
plt.savefig(loss_plot_path)
plt.show()

# ROC grafiği için gerekli hesaplamalar ve görselleştirme
from sklearn.metrics import roc_curve, auc

# Test seti için tahminlerin alınması
y_pred = model.predict(test_images)

# ROC curve için gerekli verilerin hesaplanması
fpr = {}
tpr = {}
roc_auc = {}

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# ROC grafiğinin çizilmesi
plt.figure(figsize=(10, 6))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f"ROC curve of class {i} (area = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title("Receiver Operating Characteristic")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
roc_plot_path = os.path.join(output_directory, "roc_plot.png")
plt.savefig(roc_plot_path)
plt.show()

print(f"Accuracy plot saved to --> {accuracy_plot_path}")
print(f"Loss plot saved to --> {loss_plot_path}")
print(f"ROC plot saved to --> {roc_plot_path}")
