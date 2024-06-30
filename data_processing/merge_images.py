import os
import shutil

def merge_images(source_folders, target_folder, name_prefix="iamg"):
    # Create the target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)

    image_count = 0  # Counter for image names

    for folder_name in os.listdir(source_folders[0]):  # Iterate through subfolders in the first source folder
        # Check if the subfolder exists in both source folders
        if all(os.path.isdir(os.path.join(folder, folder_name)) for folder in source_folders):
            # Create the corresponding subfolder in the target folder
            target_subfolder = os.path.join(target_folder, folder_name)
            if os.path.exists(target_subfolder):
                continue
            else:
                os.makedirs(target_subfolder, exist_ok=True)

            # Copy images from both source folders into the target subfolder with renamed names
            for source_folder in source_folders:
                source_path = os.path.join(source_folder, folder_name)
                for image_name in os.listdir(source_path):
                    image_path = os.path.join(source_path, image_name)
                    new_name = f"{name_prefix}{image_count}.jpg"  # Create new name with counter
                    target_image_path = os.path.join(target_subfolder, new_name)
                    shutil.copy(image_path, target_image_path)
                    image_count += 1  # Increment counter for next image

if __name__ == "__main__":
    source_folders = ["augmented", "images"]  # List of source folders
    target_folder = "data_images"  # Target folder name
    merge_images(source_folders, target_folder)
