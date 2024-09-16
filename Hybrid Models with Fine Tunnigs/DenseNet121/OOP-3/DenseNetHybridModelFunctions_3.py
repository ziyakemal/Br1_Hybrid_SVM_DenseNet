import os
import numpy as np
import pandas as pd
import cv2
from keras.applications import DenseNet121
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from contextlib import redirect_stdout
from keras.layers import BatchNormalization


class DenseNetHybridModel:

    # ~ init fonksiyonu, bir model nesnesi oluşturulduğunda, belirtilen çıkış dizini mevcut değilse bu dizini oluşturur.
    def __init__(self, output_directory):
        self.output_directory = output_directory
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    # ~ create_dataframe fonksiyonu, belirtilen data_path dizinindeki görüntü dosyalarının yolunu
    # ~ ve bunlara karşılık gelen etiketleri içeren bir Pandas DataFrame oluşturmaktadır.
    def create_dataframe(self, data_path):
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

    # ~ print_dataset_statistics fonksiyonu,
    # ~ bir Pandas DataFrame üzerinde bazı temel istatistiksel bilgileri ekrana yazdırarak
    # ~ veri setinin genel durumunu anlamayı sağlar.
    def print_dataset_statistics(self, df, name):
        print(f"{name} DataFrame:\n", df.head())
        print(f"{name} seti boyutları--> \n", df.shape)
        print(f"Eksik veri gözlemleri--> \n", df.isnull().sum())
        print(f"Kanser Türü Sayıları--> \n", df["label"].value_counts())

    # ~ save_statistics_to_file, verilen bir Pandas DataFrame üzerinde
    # ~ hesaplanan istatistiksel bilgileri bir dosyaya kaydetmeyi sağlar.
    def save_statistics_to_file(self, df, name, save_path):
        folder_path = os.path.dirname(save_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(f"{name} DataFrame:\n")
            f.write(df.head().to_string())
            f.write("\n\n")
            f.write(f"{name} seti boyutları--> \n{df.shape}\n\n")
            f.write(f"Eksik veri gözlemleri--> \n{df.isnull().sum().to_string()}\n\n")
            f.write(
                f"Kanser Türü Sayıları--> \n{df['label'].value_counts().to_string()}\n\n"
            )

    # ~ Bu fonksiyon, bir Pandas DataFrame'de bulunan görüntü yollarını kullanarak bu görüntüleri yükler,
    # ~ istenilen boyuta getirir ve normalleştirir. Ayrıca her görüntü için ilgili etiketi alarak,
    # ~ hem görüntüleri hem de etiketleri iki ayrı NumPy dizisi olarak döndürür.
    def load_images_from_dataframe(self, df, target_size=(224, 224)):
        images = []
        labels = []
        for index, row in df.iterrows():
            img_path = row["image_data"]
            label = row["label"]
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, target_size)
                img = img / 255.0
                images.append(img)
                labels.append(label)
        return np.array(images), np.array(labels)

    # ~ Bu fonksiyon, etiketleri binarize eder ve one-hot encoding işlemi uygular,
    # ~ yani kategorik olan etiketleri sayısal değerlere dönüştürüp,
    # ~ bu sayısal değerleri de makine öğrenmesi modellerinde kullanılabilecek şekilde
    # ~ tek-sıcaklık (one-hot encoded) vektörlere çevirir.
    def binarize_labels(self, train_df, train_labels, test_labels, valid_labels):
        # Label mapping
        label_mapping = {
            label: idx for idx, label in enumerate(train_df["label"].unique())
        }

        # Convert labels to numerical values
        train_labels_num = np.array([label_mapping[label] for label in train_labels])
        test_labels_num = np.array([label_mapping[label] for label in test_labels])
        valid_labels_num = np.array([label_mapping[label] for label in valid_labels])

        # One-hot encode the labels
        train_labels_encoded = to_categorical(
            train_labels_num, num_classes=len(label_mapping)
        )
        test_labels_encoded = to_categorical(
            test_labels_num, num_classes=len(label_mapping)
        )
        valid_labels_encoded = to_categorical(
            valid_labels_num, num_classes=len(label_mapping)
        )

        num_classes = len(label_mapping)

        return (
            train_labels_encoded,
            test_labels_encoded,
            valid_labels_encoded,
            num_classes,
        )

    # ~ create_densenet121_model DenseNet121'i bir taban model olarak kullanarak,
    # ~ belirli bir sınıflandırma problemi için uygun olacak şekilde
    # ~ özelleştirilmiş bir sinir ağı modeli oluşturur. Eklenen yeni katmanlarla birlikte,
    # ~ sınıflandırma görevini yerine getirecek şekilde tasarlanmış
    # ~ ve DenseNet121'in önceden eğitilmiş katmanları dondurulmuştur.
    def create_densenet121_model(self, input_shape, num_classes):
        base_model = DenseNet121(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation="softmax")(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        # İlk başta DenseNet'in tüm katmanlarını donduruyoruz
        for layer in base_model.layers:
            layer.trainable = False

        return model

    # ~ Bu fonksiyon, önceden eğitilmiş DenseNet121 modelinin belirli katmanlarını yeniden eğitmek
    # ~ (fine-tuning) amacıyla yapılandırır. Özellikle, modelin son katmanlarının dondurulmuş halini çözerek
    # ~ (yeniden eğitilmeye hazır hale getirerek) ve öğrenme oranını düşürerek ince ayar yapar.
    # ~ Böylece model, yeni veriler üzerinde daha hassas ve performanslı hale gelir. Sonradan eklenen katmanlar eğitilmezler.
    def fine_tune_densenet121_model(self, model, num_unfreeze_layers, learning_rate):
        # DenseNet'in son n katmanını eğitmek için çözüyoruz
        for layer in model.layers[-num_unfreeze_layers:]:
            if not isinstance(
                layer, BatchNormalization
            ):  # BN katmanlarını donduruyoruz
                layer.trainable = True

        # Modeli yeniden derliyoruz (daha düşük bir öğrenme oranı ile)
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    # ~ compile_model fonksiyonu, bir modelin derlenmesi (compile edilmesi) için gerekli olan yapılandırmayı sağlar.
    # ~ Derleme işlemi, modelin eğitimde kullanacağı optimizasyon algoritmasını, kayıp fonksiyonunu ve değerlendirme metriklerini belirler
    def compile_model(self, model, learning_rate=0.001):
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    # ~ Bu fonksiyon, bir derin öğrenme modelini verilen eğitim ve doğrulama verileriyle eğitir
    # ~ ve modelin eğitim sürecinin geçmişini (history) döndürür. Eğitim süreci boyunca,
    # ~ modelin performansı her bir epoch'ta (eğitim döngüsü) güncellenir ve isteğe bağlı olarak
    # ~ callbacks kullanılarak ek işlemler yapılabilir (örneğin, model ağırlıklarının kaydedilmesi, erken durdurma, vb.).
    def train_model(
        self,
        model,
        train_images,
        train_labels,
        valid_images,
        valid_labels,
        epochs,
        callbacks,
    ):
        history = model.fit(
            train_images,
            train_labels,
            validation_data=(valid_images, valid_labels),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
        )
        return history

    # ~ create_callbacks fonksiyonu, derin öğrenme modelini eğitirken kullanılan callback fonksiyonlarını oluşturur ve döndürür.
    # ~ Callback fonksiyonları, eğitim sürecine müdahale eden, modelin performansını izleyen ve eğitim sırasında belirli olayları
    # ~ (örneğin, erken durdurma, modelin ağırlıklarını kaydetme, öğrenme oranını ayarlama gibi) tetikleyen yardımcı araçlardır.
    def create_callbacks(
        self,
        train_df_shape,
        checkpoint_name="epoch_{epoch:02d}_val_acc_{val_accuracy:.2f}.hdf5",
        prefix="",
        patience=5,
        min_lr=1e-6,
    ):
        ckp_interval = 5 * int(np.ceil(train_df_shape / 64))
        ckp_path = os.path.join(self.output_directory, prefix + checkpoint_name)
        checkpoint = ModelCheckpoint(
            filepath=ckp_path,
            save_weights_only=True,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
        )
        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=patience, min_lr=min_lr
            ),
            checkpoint,
        ]
        return callbacks

    # ~ Bu fonksiyon, modelin eğitim sürecinde elde edilen history (eğitim geçmişi) bilgisini bir dosyaya kaydeder.
    # ~ Eğitim geçmişi, modelin her epoch'ta kaydettiği doğruluk (accuracy), kayıp (loss), doğrulama doğruluğu (val_accuracy),
    # ~ doğrulama kaybı (val_loss) gibi bilgileri içerir. Bu bilgiler, modelin performansını analiz etmek için daha sonra kullanılabilir.
    def save_history(self, history, filename):
        history_file_path = os.path.join(self.output_directory, filename)
        with open(history_file_path, "wb") as f:
            np.save(f, history.history)
        print(f"History saved to --> {history_file_path}")

    # ~ Bu fonksiyon, bir numpy dizisi kullanarak eğitim veri setinin istatistiksel özetlerini belirtilen bir dosyaya kaydeder.
    # ~ Görüntülerin boyutları, etiketlerin şekli, benzersiz etiketler ve etiketlerin dağılımı gibi temel bilgileri kaydeder.
    def save_numpy_statistics(self, images, labels, save_path, dataset_name):
        folder_path = os.path.dirname(save_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(f"{dataset_name} Dataset Betimsel İstatistikler:\n")
            f.write(f"Images shape: {images.shape}\n")
            f.write(f"Labels shape: {labels.shape}\n")
            f.write(f"Unique Labels: {np.unique(np.argmax(labels, axis=1))}\n")
            f.write(f"Labels distribution:\n{np.sum(labels, axis=0)}\n")

    # ~ Bu fonksiyon, verilen bir derin öğrenme modelinin özetini (modelin yapısını) bir dosyaya kaydetmek için yazılmıştır.
    # ~ Model özeti, modeldeki katmanları, her katmanın çıkış şekillerini ve modeldeki toplam parametre sayısını içerir.
    # ~ Bu özet genellikle modelin yapısına dair önemli bir genel bakış sağlar.
    def save_model_summary(self, model, summary_file_name):
        summary_file_path = os.path.join(self.output_directory, summary_file_name)
        with open(summary_file_path, "w") as f:
            with redirect_stdout(f):
                model.summary()
        print(f"Model summary saved to --> {summary_file_path}")

    # ~ Bu fonksiyon, bir derin öğrenme modelini belirli bir dosya yoluna kaydetmek için kullanılır.
    # ~ Modelin yapısı, ağırlıkları ve eğitilmiş hali bu dosyaya kaydedilir, böylece model daha sonra tekrar yüklenip kullanılabilir.
    # ~ Bu, modelin eğitimi tamamlandıktan sonra, ileride yeniden eğitime gerek kalmadan aynı modelin kullanılabilmesini sağlar.
    def save_model(self, model, model_save_name):
        model_save_path = os.path.join(self.output_directory, model_save_name)
        model.save(model_save_path)
        print(f"Model saved to --> {model_save_path}")

    # ~ Bu fonksiyon, bir derin öğrenme modelinin test veri seti üzerindeki performansını değerlendirir ve doğruluk (accuracy) ile
    # ~ kayıp (loss) değerlerini döndürür. Bu tür bir değerlendirme işlemi, modelin daha önce görmediği test verisi üzerindeki
    # ~ genel performansını ölçmek için kullanılır.
    def evaluate_model(self, model, test_images, test_labels):
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print(f"Test Accuracy: {test_acc}")
        return test_loss, test_acc
