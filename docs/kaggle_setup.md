# Kaggle API Setup Guide

## How to Get Your Kaggle API Credentials

### Method 1: From Kaggle Website (Recommended)

1. **Go to Kaggle**: Visit [https://www.kaggle.com/](https://www.kaggle.com/)

2. **Sign in or Create Account**: 
   - If you don't have an account, create one
   - If you have an account, sign in

3. **Go to Account Settings**:
   - Click on your profile picture (top right)
   - Select "Account" from the dropdown menu
   - Or go directly to: [https://www.kaggle.com/settings/account](https://www.kaggle.com/settings/account)

4. **Create New API Token**:
   - Scroll down to the "API" section
   - Click "Create New API Token"
   - This will download a `kaggle.json` file to your computer

5. **Save the File**: 
   - The `kaggle.json` file contains your credentials
   - Keep this file secure and don't share it publicly

### Method 2: Direct Download Link

If you prefer not to use Kaggle API, you can download the dataset directly:

**PlantVillage Dataset Sources:**
- **Kaggle**: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
- **Original GitHub**: https://github.com/spMohanty/PlantVillage-Dataset
- **Papers with Code**: https://paperswithcode.com/dataset/plantvillage

## Using kaggle.json in Google Colab

### Option A: Upload During Runtime
```python
from google.colab import files
uploaded = files.upload()  # Upload your kaggle.json file
```

### Option B: Store in Google Drive
1. Upload `kaggle.json` to your Google Drive
2. In Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
!cp '/content/drive/MyDrive/kaggle.json' ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
```

### Option C: Manual Entry
```python
import json
import os

# Create kaggle directory
os.makedirs('/root/.kaggle', exist_ok=True)

# Enter your credentials
kaggle_credentials = {
    "username": "your_kaggle_username",
    "key": "your_kaggle_key"
}

# Save credentials
with open('/root/.kaggle/kaggle.json', 'w') as f:
    json.dump(kaggle_credentials, f)

# Set permissions
os.chmod('/root/.kaggle/kaggle.json', 0o600)
```

## Alternative: No Kaggle Account Needed

If you don't want to create a Kaggle account, use our alternative download script:

```python
# Run our alternative download script
!python data/download_dataset.py
```

This script provides multiple download options:
1. Google Drive mirror
2. Direct download links
3. Manual upload instructions
4. Sample dataset creation

## Dataset Information

**PlantVillage Dataset Details:**
- **Size**: ~1.5 GB
- **Images**: 54,306 images
- **Classes**: 38 disease categories
- **Plants**: Apple, Corn, Tomato, and more
- **Format**: RGB images (various sizes)

**Disease Categories Include:**
- Apple: Apple scab, Black rot, Cedar apple rust, Healthy
- Corn: Cercospora leaf spot, Common rust, Northern Leaf Blight, Healthy
- Tomato: Bacterial spot, Early blight, Late blight, Healthy
- And many more...

## Troubleshooting

### Common Issues:

1. **"API credentials not found"**:
   - Make sure kaggle.json is in the right location
   - Check file permissions: `chmod 600 ~/.kaggle/kaggle.json`

2. **"Dataset not found"**:
   - Verify the dataset URL
   - Check if the dataset is public

3. **"Permission denied"**:
   - Accept the dataset terms on Kaggle website
   - Join the competition/dataset if required

4. **Download fails**:
   - Check internet connection
   - Try alternative download methods
   - Use our backup download script

### Alternative Solutions:

1. **Manual Download**:
   - Download from Kaggle website manually
   - Upload to Google Drive
   - Access from Colab using Drive mount

2. **Use Our Script**:
   ```bash
   python data/download_dataset.py
   ```

3. **Contact Support**:
   - If all methods fail, create an issue on GitHub
   - We can provide additional download mirrors

## Security Note

⚠️ **Important**: Never commit your `kaggle.json` file to version control or share it publicly. It contains your API credentials.
