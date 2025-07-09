#!/usr/bin/env python3
"""
Alternative dataset download script for PlantVillage dataset.
This script provides multiple methods to download the dataset.
"""

import os
import requests
import zipfile
from pathlib import Path
import gdown
from tqdm import tqdm

def download_file(url, filename):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))

def download_from_google_drive():
    """Download dataset from Google Drive mirror."""
    print("📥 Downloading PlantVillage dataset from Google Drive...")
    
    # Create data directory
    os.makedirs('data/plant_village/raw', exist_ok=True)
    
    # Google Drive file ID for PlantVillage dataset
    file_id = "1aFLdWFMQXZX7HvgwfmIE6KqgOBGGBhQ3"  # This is an example ID
    url = f"https://drive.google.com/uc?id={file_id}"
    
    try:
        # Download using gdown
        output_path = "data/plant_village/raw/plant_village_dataset.zip"
        gdown.download(url, output_path, quiet=False)
        
        # Extract
        print("📦 Extracting dataset...")
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall('data/plant_village/raw/')
        
        # Clean up zip file
        os.remove(output_path)
        print("✅ Dataset downloaded and extracted successfully!")
        
    except Exception as e:
        print(f"❌ Error downloading from Google Drive: {e}")
        return False
    
    return True

def download_from_direct_link():
    """Download from direct link (if available)."""
    print("📥 Downloading PlantVillage dataset from direct link...")
    
    # Create data directory
    os.makedirs('data/plant_village/raw', exist_ok=True)
    
    # Direct download URLs (you may need to update these)
    urls = [
        "https://data.mendeley.com/datasets/tywbtsjrjv/1/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/PlantVillage.zip",
        # Add more alternative URLs here
    ]
    
    for url in urls:
        try:
            print(f"Trying: {url}")
            output_path = "data/plant_village/raw/plant_village_dataset.zip"
            
            # Download
            download_file(url, output_path)
            
            # Extract
            print("📦 Extracting dataset...")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall('data/plant_village/raw/')
            
            # Clean up zip file
            os.remove(output_path)
            print("✅ Dataset downloaded and extracted successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download from {url}: {e}")
            continue
    
    return False

def create_sample_dataset():
    """Create a small sample dataset for testing."""
    print("🔬 Creating sample dataset for testing...")
    
    # Create directory structure
    sample_dir = Path('data/plant_village/raw/sample_dataset')
    
    # Create some sample disease categories
    diseases = [
        'Apple___Apple_scab',
        'Apple___Black_rot',
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___healthy'
    ]
    
    for disease in diseases:
        disease_dir = sample_dir / disease
        disease_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a placeholder file
        placeholder_file = disease_dir / 'README.md'
        with open(placeholder_file, 'w') as f:
            f.write(f"# {disease}\n\nThis is a placeholder for {disease} images.\n")
    
    print("✅ Sample dataset structure created!")
    print("📝 You can now manually add images to each disease folder.")
    return True

def main():
    print("🌱 PlantVillage Dataset Download Options")
    print("=" * 50)
    
    methods = [
        ("1", "Download from Google Drive", download_from_google_drive),
        ("2", "Download from Direct Link", download_from_direct_link),
        ("3", "Create Sample Dataset (for testing)", create_sample_dataset),
        ("4", "Manual Upload Instructions", show_manual_instructions)
    ]
    
    print("\nAvailable download methods:")
    for key, desc, _ in methods:
        print(f"  {key}. {desc}")
    
    choice = input("\nChoose a method (1-4): ").strip()
    
    method_map = {key: func for key, _, func in methods}
    
    if choice in method_map:
        success = method_map[choice]()
        if success:
            print("\n🎉 Dataset setup complete!")
            print("📝 Next step: Run data preprocessing with:")
            print("    python data/data_preprocessing.py")
        else:
            print("\n❌ Dataset download failed. Try another method.")
    else:
        print("❌ Invalid choice. Please run the script again.")

def show_manual_instructions():
    """Show instructions for manual dataset upload."""
    print("\n📋 Manual Dataset Upload Instructions")
    print("=" * 40)
    
    print("\n🔗 Dataset Sources:")
    print("1. Kaggle: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
    print("2. Original: https://github.com/spMohanty/PlantVillage-Dataset")
    print("3. Papers with Code: https://paperswithcode.com/dataset/plantvillage")
    
    print("\n📁 Manual Upload Steps:")
    print("1. Download the dataset from any of the above sources")
    print("2. Extract the zip file")
    print("3. Copy the extracted folder to: data/plant_village/raw/")
    print("4. Ensure the structure looks like:")
    print("   data/plant_village/raw/")
    print("   ├── Plant_leave_diseases_dataset_without_augmentation/")
    print("   │   ├── Apple___Apple_scab/")
    print("   │   ├── Apple___Black_rot/")
    print("   │   └── ... (other disease folders)")
    print("   └── (or similar structure)")
    
    print("\n💡 For Google Colab:")
    print("1. Upload the dataset zip file to your Google Drive")
    print("2. Use this code in Colab:")
    print("   from google.colab import drive")
    print("   drive.mount('/content/drive')")
    print("   !unzip '/content/drive/MyDrive/your_dataset.zip' -d data/plant_village/raw/")
    
    return True

if __name__ == "__main__":
    main()
