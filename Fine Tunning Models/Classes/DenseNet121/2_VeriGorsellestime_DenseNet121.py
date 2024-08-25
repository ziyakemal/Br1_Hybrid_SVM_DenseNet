#
# * 1-) --------------------------------------------------------------------------------------------------------
# Kütüphanelerimizi aşağıdaki gibi import edelim.


# For Data Visualization
import matplotlib.pyplot as plt


# Miscellaneous
import cv2
import os


# & ______________________________________ Görselleştirme Fonksiyonu ___________________________________________
# * 2-) --------------------------------------------------------------------------------------------------------


import os
import cv2
import matplotlib.pyplot as plt


def visualize_images(image_path, title, save_name, save_dir):
    # Save directory exists check and create if not
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Full save path
    full_save_path = os.path.join(save_dir, save_name)

    image_files = os.listdir(image_path)
    plt.figure(figsize=(10, 10))
    for i in range(25):
        img_path = os.path.join(image_path, image_files[i])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(5, 5, i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(full_save_path)
    plt.close()
    print(f"Saved -- > {full_save_path}")


# Örnek kullanımı:
image_folder = "dataSet/NO"  # Görüntülerin olduğu klasör
save_directory = (
    "Fine Tunning Models/Figures And Tables/DenseNet121"  # Kayıt edilecek klasör yolu
)
save_filename = "no_image_grid.png"  # Kayıt edilecek dosya adı

visualize_images(image_folder, "Eğitim Verileri", save_filename, save_directory)


# & ____________________________ Train ve Test Setlerindeki Kanser Dağılımları _________________________________
# * 3-) --------------------------------------------------------------------------------------------------------

import os
import matplotlib.pyplot as plt


def create_pie_chart(ax, data, title, colors, explode, counts):
    wedges, texts, autotexts = ax.pie(
        data.values(),
        labels=data.keys(),
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        explode=explode,
    )
    ax.axis("equal")
    ax.set_title(title, weight="bold")

    # Adding the count text below the pie chart
    ax.text(
        0.5,
        -0.1,
        f"Yes: {counts['Yes']}\nNo: {counts['No']}",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=12,
    )


# Define the path for each dataset
data_path = "dataSet_Final"
subfolders = ["Train", "Test", "Validation"]

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Loop through each subfolder to create pie charts
for i, subfolder in enumerate(subfolders):
    # Get the paths for "Yes" and "No" folders
    yes_path = os.path.join(data_path, subfolder, "Yes")
    no_path = os.path.join(data_path, subfolder, "No")

    # Count the number of images in each folder
    yes_count = len(os.listdir(yes_path))
    no_count = len(os.listdir(no_path))

    # Create the data dictionary
    label_counts = {"Yes": yes_count, "No": no_count}
    total_count = yes_count + no_count
    label_percentages = {
        label: count / total_count * 100 for label, count in label_counts.items()
    }

    # Define colors and explode settings
    colors = ["seagreen", "lightcoral"]
    explode = [0, 0.1]  # "No" class should be exploded

    # Create the pie chart with counts
    create_pie_chart(
        axes[i],
        label_percentages,
        f"{subfolder} Data Distribution",
        colors,
        explode,
        label_counts,  # Pass the counts to display under the pie chart
    )

plt.tight_layout()

# Define the directory to save the image
save_directory = "Fine Tunning Models/Figures And Tables/DenseNet121"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Save the image
save_path = os.path.join(save_directory, "DataSet_Distribution.png")
plt.savefig(save_path)
print(f"Saved -- > {save_path}")
plt.close()


# import os
# import matplotlib.pyplot as plt


# def create_pie_chart(ax, data, title, colors, explode):
#     ax.pie(
#         data.values(),
#         labels=data.keys(),
#         autopct="%1.1f%%",
#         startangle=140,
#         colors=colors,
#         explode=explode,
#     )
#     ax.axis("equal")
#     ax.set_title(title, weight="bold")


# # Define the path for each dataset
# data_path = "dataSet_Final"
# subfolders = ["Train", "Test", "Validation"]

# # Create subplots
# fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# # Loop through each subfolder to create pie charts
# for i, subfolder in enumerate(subfolders):
#     # Get the paths for "Yes" and "No" folders
#     yes_path = os.path.join(data_path, subfolder, "Yes")
#     no_path = os.path.join(data_path, subfolder, "No")

#     # Count the number of images in each folder
#     yes_count = len(os.listdir(yes_path))
#     no_count = len(os.listdir(no_path))

#     # Create the data dictionary
#     label_counts = {"Yes": yes_count, "No": no_count}
#     total_count = yes_count + no_count
#     label_percentages = {
#         label: count / total_count * 100 for label, count in label_counts.items()
#     }

#     # Define colors and explode settings
#     colors = ["seagreen", "lightcoral"]
#     explode = [0, 0.1]  # "No" class should be exploded

#     # Create the pie chart
#     create_pie_chart(
#         axes[i],
#         label_percentages,
#         f"{subfolder} Data Distribution",
#         colors,
#         explode,
#     )

# plt.tight_layout()

# # Define the directory to save the image
# save_directory = "Fine Tunning Models/Figures And Tables/DenseNet121"
# if not os.path.exists(save_directory):
#     os.makedirs(save_directory)

# # Save the image
# save_path = os.path.join(save_directory, "DataSet_Distribution.png")
# plt.savefig(save_path)
# print(f"Saved -- > {save_path}")
# plt.close()
