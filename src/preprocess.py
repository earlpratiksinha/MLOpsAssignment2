import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def clean_and_split():
    categories = ["Cat", "Dog"]
    
    for category in categories:
        raw_cat_dir = os.path.join(RAW_DIR, category)
        valid_images = []
        
        # Verify non-corrupted images
        for img_name in os.listdir(raw_cat_dir):
            img_path = os.path.join(raw_cat_dir, img_name)
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            try:
                with Image.open(img_path) as img:
                    img.verify()
                valid_images.append(img_path)
            except Exception:
                print(f"Skipping corrupted image: {img_path}")

        # Split into train (70%), val (15%), test (15%)
        train_imgs, test_imgs = train_test_split(valid_images, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(test_imgs, test_size=0.5, random_state=42)

        splits = {'train': train_imgs, 'val': val_imgs, 'test': test_imgs}
        
        for split_name, img_list in splits.items():
            split_dir = os.path.join(PROCESSED_DIR, split_name, category)
            os.makedirs(split_dir, exist_ok=True)
            for img_path in img_list:
                shutil.copy(img_path, os.path.join(split_dir, os.path.basename(img_path)))

    print("Data preprocessing and splitting complete!")

if __name__ == "__main__":
    clean_and_split()
