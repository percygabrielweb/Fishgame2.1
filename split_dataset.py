from pathlib import Path

def split_images_into_folders(source_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits the images in the source folder into three separate folders: train, val, and test, based on the provided ratios.
    """
    # Ensure that the sum of the ratios is 1.0
    if (train_ratio + val_ratio + test_ratio) != 1.0:
        raise ValueError("The sum of the ratios must be 1.0")

    source_folder_path = Path(source_folder)

    # Create train, val, and test folders
    train_folder = source_folder_path / "train"
    val_folder = source_folder_path / "val"
    test_folder = source_folder_path / "test"

    for folder in [train_folder, val_folder, test_folder]:
        folder.mkdir(exist_ok=True)

    # List all images in the source folder and sort them
    image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.txt']
    all_images = sorted([img for img in source_folder_path.iterdir() if img.suffix.lower() in image_extensions], key=lambda x: x.name)

    total_images = len(all_images)

    # Calculate the number of images for train, val, and test
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)
    test_count = total_images - train_count - val_count

    # Move images to their respective folders
    for i, img in enumerate(all_images):
        if i < train_count:
            img.rename(train_folder / img.name)
        elif i < (train_count + val_count):
            img.rename(val_folder / img.name)
        else:
            img.rename(test_folder / img.name)

if __name__ == "__main__":
    source_folder_path = 'dataset/labels'  # Change this to the path of your source folder
    split_images_into_folders(source_folder_path)
    source_folder_path = 'dataset/images'  # Change this to the path of your source folder
    split_images_into_folders(source_folder_path)
