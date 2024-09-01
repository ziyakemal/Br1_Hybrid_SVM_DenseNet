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


class DenseNetHybridModel:
    def __init__(self, output_directory):
        self.output_directory = output_directory
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

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

    def print_dataset_statistics(self, df, name):
        print(f"{name} DataFrame:\n", df.head())
        print(f"{name} seti boyutları--> \n", df.shape)
        print(f"Eksik veri gözlemleri--> \n", df.isnull().sum())
        print(f"Kanser Türü Sayıları--> \n", df["label"].value_counts())

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

    # def binarize_labels(self, df, labels):
    #     label_mapping = {label: idx for idx, label in enumerate(df["label"].unique())}
    #     binarized_labels = np.array([label_mapping[label] for label in labels])
    #     binarized_labels = to_categorical(binarized_labels, num_classes=len(label_mapping))
    #     return binarized_labels, label_mapping

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

    def create_densenet121_model(
        self, input_shape, num_classes, trainable_layers=0, learning_rate=0.0001
    ):
        base_model = DenseNet121(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        for layer in base_model.layers[-trainable_layers:]:
            layer.trainable = True
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

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

    def save_history(self, history, filename):
        history_file_path = os.path.join(self.output_directory, filename)
        with open(history_file_path, "wb") as f:
            np.save(f, history.history)
        print(f"History saved to --> {history_file_path}")

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

    def save_model_summary(self, model, summary_file_name):
        summary_file_path = os.path.join(self.output_directory, summary_file_name)
        with open(summary_file_path, "w") as f:
            with redirect_stdout(f):
                model.summary()
        print(f"Model summary saved to --> {summary_file_path}")

    def save_model(self, model, model_save_name):
        model_save_path = os.path.join(self.output_directory, model_save_name)
        model.save(model_save_path)
        print(f"Model saved to --> {model_save_path}")

    def evaluate_model(self, model, test_images, test_labels):
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print(f"Test Accuracy: {test_acc}")
        return test_loss, test_acc
