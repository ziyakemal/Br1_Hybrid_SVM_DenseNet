""" from keras.models import load_model

# En son kaydedilen modeli yükleyin
model = load_model('C:/Users/ziyak/Desktop/Asus ExpertBook Tum Yedek/OneDrive/Documents/GitHub/MultipleSclerosis_AI/ModelCheckPoints_MS/epocch_07_val_acc_0.86.hdf5')

history = model.fit(
    train_data_generator,
    batch_size=batch_size,
    validation_data=valid_data_generator,
    epochs=50,  # Toplam tamamlamak istediğiniz epoch sayısı
    initial_epoch=7,  # Eğitimin kaldığı epoch
    callbacks=callbacks,
)

model.save("MS_classifier_MS_continued.keras")

# History verisini bir dosyaya kaydedin
with open("history_MS_continued.npy", "wb") as f:
    np.save(f, history.history)
 """
