import os
import random
import shutil


def move_random_files(source_dir, dest_dir, percentage):
    # Kaynak klasördeki tüm JPEG dosyalarını listele
    jpeg_files = [f for f in os.listdir(source_dir) if f.endswith(".jpeg")]

    # Taşınacak dosya sayısını hesapla
    num_files_to_move = int(len(jpeg_files) * (percentage / 100))

    # Rastgele dosyalar seç
    files_to_move = random.sample(jpeg_files, num_files_to_move)

    # Dosyaları taşı
    for file_name in files_to_move:
        source_file = os.path.join(source_dir, file_name)
        dest_file = os.path.join(dest_dir, file_name)
        shutil.move(source_file, dest_file)

    print(f"{num_files_to_move} dosya başarıyla taşındı.")


# Kullanım
source_directory = "dataSet_Aug_Copy/YES_Aug"
destination_directory = "dataSet_Final/Validation/Yes"
percentage_to_move = 70  # Örneğin %20

move_random_files(source_directory, destination_directory, percentage_to_move)
