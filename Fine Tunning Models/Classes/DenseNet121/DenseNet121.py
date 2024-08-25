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


# Miscellaneous
import cv2

# from tqdm import tqdm
import os
import random
import pickle

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
