from DenseNetHybridModelFunctions_2 import DenseNetHybridModel

from keras.optimizers import Adam

output_directory = "Hybrid Models with Fine Tunnigs/DenseNet121/Models And History"


class DenseNet121_HybridModel_Test:
    def __init__(self, output_directory):
        # DenseNetHybridModel sınıfından bir instance oluşturuyoruz ve 'model_instance' değişkenine atıyoruz
        self.model_instance = DenseNetHybridModel(output_directory)

    def createDataFrame(self, data_path):
        df = self.model_instance.create_dataframe(data_path)
        return df

    def printDataSetStatistics(self, df, name):
        self.model_instance.print_dataset_statistics(df, name)

    def saveStatisticsToFile(self, df, name, save_path):
        self.model_instance.save_statistics_to_file(df, name, save_path)

    def loadImagesFromDataframe(self, df, target_size=(224, 224)):
        return self.model_instance.load_images_from_dataframe(df, target_size)

    def binarizeLabels(self, train_df, train_labels, test_labels, valid_labels):
        return self.model_instance.binarize_labels(
            train_df, train_labels, test_labels, valid_labels
        )

    def createDensenet121Model(self, input_shape, num_classes):
        return self.model_instance.create_densenet121_model(input_shape, num_classes)

    def saveModelSummary(self, model, summary_file_name):
        return self.model_instance.save_model_summary(model, summary_file_name)

    def createCallbacks(
        self,
        train_df_shape,
        checkpoint_name="epoch_{epoch:02d}_val_acc_{val_accuracy:.2f}.hdf5",
        prefix="",
        patience=5,
        min_lr=1e-6,
    ):
        return self.model_instance.create_callbacks(
            train_df_shape,
            checkpoint_name,
            prefix,
            patience,
            min_lr,
        )

    def trainModel(
        self,
        model,
        train_images,
        train_labels,
        valid_images,
        valid_labels,
        epochs,
        callbacks,
    ):
        return self.model_instance.train_model(
            model,
            train_images,
            train_labels,
            valid_images,
            valid_labels,
            epochs,
            callbacks,
        )

    def saveHistory(self, history, filename):
        return self.model_instance.save_history(history, filename)

    def saveModel(self, model, model_save_name):
        return self.model_instance.save_model(model, model_save_name)

    def saveNumpyStatistics(self, images, labels, save_path, dataset_name):
        return self.model_instance.save_numpy_statistics(
            images, labels, save_path, dataset_name
        )

    def evaluateModel(self, model, test_images, test_labels):
        return self.model_instance.evaluate_model(model, test_images, test_labels)

    def compileModel(self, model, learning_rate=0.0001):
        return self.model_instance.compile_model(model, learning_rate)


# & ____________________________________ Instance Oluştur __________________________________________
# * 1-) --------------------------------------------------------------------------------------------------------
denseNet121_HybridModel_Test = DenseNet121_HybridModel_Test(output_directory)


# & ____________________________________ DataFramelerin Elde Edilmesi __________________________________________
# * 2-) --------------------------------------------------------------------------------------------------------

train_data = "dataSet_Final/Train"
test_data = "dataSet_Final/Test"
valid_data = "dataSet_Final/Validation"

train_df = denseNet121_HybridModel_Test.createDataFrame(train_data)
test_df = denseNet121_HybridModel_Test.createDataFrame(test_data)
valid_df = denseNet121_HybridModel_Test.createDataFrame(valid_data)

# & ___________________________ Betimsel İstatistiklerin Elde Edilmesi _________________________________
# * 3-) --------------------------------------------------------------------------------------------------------

denseNet121_HybridModel_Test.printDataSetStatistics(train_df, "Eğitim")
denseNet121_HybridModel_Test.printDataSetStatistics(valid_df, "Validasyon")
denseNet121_HybridModel_Test.printDataSetStatistics(test_df, "Test")

# * 4-) --------------------------------------------------------------------------------------------------------
# Dataset istatistiklerinin txt dosyası olarak kayıt edilmesi.


denseNet121_HybridModel_Test.saveStatisticsToFile(
    valid_df,
    "Validasyon",
    "Hybrid Models with Fine Tunnigs/DenseNet121/Figure And Tables/validationStatistic.txt",
)

denseNet121_HybridModel_Test.saveStatisticsToFile(
    valid_df,
    "Eğitim",
    "Hybrid Models with Fine Tunnigs/DenseNet121/Figure And Tables/trainStatistic.txt",
)

denseNet121_HybridModel_Test.saveStatisticsToFile(
    valid_df,
    "Test",
    "Hybrid Models with Fine Tunnigs/DenseNet121/Figure And Tables/testStatistic.txt",
)

# & _________________________ Görüntülerin yüklenmesi ve DataFrame'den NumPy dizilerine dönüştürülmesi _______________________________
# * 5-) ------------------------------------------------------------------------------------------------------------------------------

# Verilerin yüklenmesi
train_images, train_labels = denseNet121_HybridModel_Test.loadImagesFromDataframe(
    train_df
)
test_images, test_labels = denseNet121_HybridModel_Test.loadImagesFromDataframe(test_df)
valid_images, valid_labels = denseNet121_HybridModel_Test.loadImagesFromDataframe(
    valid_df
)

# & ___________________________________ Etiketlerin Binarize Edilmesi _____________________________________________
# * 6-) -----------------------------------------------------------------------------------------------------------

train_labels_encoded, test_labels_encoded, valid_labels_encoded, num_classes = (
    denseNet121_HybridModel_Test.binarizeLabels(
        train_df, train_labels, test_labels, valid_labels
    )
)

# & _________________________________________ Eğitim Sürecini Uygulama ________________________________________________
# * 7-) --------------------------------------------------------------------------------------------------------

# ~ Modelin oluşturulması ve derlenmesi (Son katmanlar eğitilecek)
image_shape = (224, 224, 3)

model = denseNet121_HybridModel_Test.createDensenet121Model(
    input_shape=image_shape, num_classes=num_classes
)

denseNet121_HybridModel_Test.compileModel(model, learning_rate=0.0001)

# ~ Model özetini kaydetme
denseNet121_HybridModel_Test.saveModelSummary(
    model, "DenseNet121_SVM_HybridModel_summary.txt"
)

# ~ Callback'lerin oluşturulması
callbacks = denseNet121_HybridModel_Test.createCallbacks(
    train_df.shape[0], prefix="pretrain_"
)

# ~ Ön Eğitim: Sadece son katmanlar eğitilir
history_pretrain = denseNet121_HybridModel_Test.trainModel(
    model,
    train_images,
    train_labels_encoded,
    valid_images,
    valid_labels_encoded,
    epochs=10,
    callbacks=callbacks,
)

# ~ Fine-Tuning: Son 30 katman da eğitilebilir hale getiriliyor
model = denseNet121_HybridModel_Test.createDensenet121Model(
    input_shape=image_shape, num_classes=num_classes
)

for layer in model.layers[-30:]:  # Son 30 katman eğitilebilir hale getirildi
    layer.trainable = True

denseNet121_HybridModel_Test.compileModel(model, learning_rate=0.00001)

# ~ Callback'lerin oluşturulması
callbacks = denseNet121_HybridModel_Test.createCallbacks(
    train_df.shape[0], prefix="fineTunning_"
)

# ~ Fine-Tuning Eğitimi
history_finetune = denseNet121_HybridModel_Test.trainModel(
    model,
    train_images,
    train_labels_encoded,
    valid_images,
    valid_labels_encoded,
    epochs=20,
    callbacks=callbacks,
)

# & ___________________________ Model ve Geçmişin Kaydedilmesi ______________________________________
# * 8-) ------------------------------------------------------------------------------------------------

denseNet121_HybridModel_Test.saveHistory(
    history_pretrain, "history_DenseNet121_pretrain.npy"
)
denseNet121_HybridModel_Test.saveHistory(
    history_finetune, "history_DenseNet121_fineTunning.npy"
)

# ~ Modelin kaydedilmesi
denseNet121_HybridModel_Test.saveModel(model, "DenseNet121_classifier.keras")

# ~ +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Modelin test edilmesi
denseNet121_HybridModel_Test.evaluateModel(model, test_images, test_labels_encoded)

# & ___________________________ Tüm Datasetlerin Betimsel İstatistiklerini Kaydet ___________________________________________
# * 9-) ------------------------------------------------------------------------------------------------

# Train seti için
train_images_stats_path = "Hybrid Models with Fine Tunnigs/DenseNet121/Figure And Tables/train_images_statistics.txt"
denseNet121_HybridModel_Test.saveNumpyStatistics(
    train_images, train_labels_encoded, train_images_stats_path, "Train"
)

# Validation seti için
valid_images_stats_path = "Hybrid Models with Fine Tunnigs/DenseNet121/Figure And Tables/valid_images_statistics.txt"
denseNet121_HybridModel_Test.saveNumpyStatistics(
    valid_images, valid_labels_encoded, valid_images_stats_path, "Validation"
)

# Test seti için
test_images_stats_path = "Hybrid Models with Fine Tunnigs/DenseNet121/Figure And Tables/test_images_statistics.txt"
denseNet121_HybridModel_Test.saveNumpyStatistics(
    test_images, test_labels_encoded, test_images_stats_path, "Test"
)
