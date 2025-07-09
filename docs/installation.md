# Installation Guide

This guide will help you set up the plant disease detection project on your local machine.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional but recommended for training)
- Git

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/plant-disease-detection-vit.git
cd plant-disease-detection-vit
```

### 2. Create Virtual Environment

It's recommended to use a virtual environment to avoid conflicts with other projects.

```bash
# Using conda
conda create -n plant-disease python=3.9
conda activate plant-disease

# Or using venv
python -m venv plant-disease-env
source plant-disease-env/bin/activate  # On Windows: plant-disease-env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the Package

For development installation:

```bash
pip install -e .
```

For regular installation:

```bash
pip install .
```

## GPU Setup (Optional)

If you have a CUDA-compatible GPU, you can install the GPU version of PyTorch for faster training:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Verify Installation

Run the following command to verify that everything is installed correctly:

```bash
python -c "import torch; import torchvision; import transformers; print('All dependencies installed successfully!')"
```

## Download Dataset

The project uses the PlantVillage dataset. You can download it from:

1. [Kaggle PlantVillage Dataset](https://www.kaggle.com/emmarex/plantdisease)
2. [Official PlantVillage Website](https://plantvillage.psu.edu/)

After downloading, extract the dataset and place it in the `data/plant_village/raw/` directory.

## Data Preprocessing

Once you have the raw dataset, preprocess it by running:

```bash
python data/data_preprocessing.py
```

This will split the dataset into train/validation/test sets and save them in `data/plant_village/processed/`.

## Running Tests

To ensure everything is working correctly, run the unit tests:

```bash
python -m pytest tests/
```

Or run individual test files:

```bash
python tests/test_models.py
python tests/test_data_loading.py
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in training scripts
2. **Module not found errors**: Make sure you're in the project root directory and have installed the package
3. **Slow training on CPU**: Consider using a smaller model or getting access to a GPU

### Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/yourusername/plant-disease-detection-vit/issues)
2. Read the [Usage Guide](usage.md)
3. Consult the [API Reference](api_reference.md)
