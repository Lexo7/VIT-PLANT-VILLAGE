# Plant Disease Detection with Vision Transformers

A comprehensive project for detecting plant diseases using Convolutional Neural Networks (CNNs), Vision Transformers (ViTs), and hybrid approaches.

## Project Structure

```
plant-disease-detection-vit/
├── README.md
├── requirements.txt
├── setup.py
├── data/
│   ├── plant_village/
│   └── data_preprocessing.py
├── models/
│   ├── __init__.py
│   ├── cnn_models.py
│   ├── vit_models.py
│   └── hybrid_models.py
├── training/
│   ├── train_cnn.py
│   ├── train_vit.py
│   └── train_hybrid.py
├── evaluation/
│   ├── evaluate_models.py
│   ├── visualize_attention.py
│   └── generate_reports.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_training.ipynb
│   ├── 03_vit_implementation.ipynb
│   └── 04_hybrid_experiments.ipynb
├── results/
│   ├── model_weights/
│   ├── evaluation_results/
│   └── visualizations/
├── docs/
│   ├── installation.md
│   ├── usage.md
│   └── api_reference.md
└── tests/
    ├── test_models.py
    └── test_data_loading.py
```

## Overview

This project explores different approaches to plant disease detection:
- Traditional CNN models for baseline performance
- Vision Transformers (ViTs) for advanced feature extraction
- Hybrid models combining CNNs and ViTs

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download and prepare data:
   ```bash
   python data/data_preprocessing.py
   ```

3. Train models:
   ```bash
   # Train CNN baseline
   python training/train_cnn.py
   
   # Train Vision Transformer
   python training/train_vit.py
   
   # Train hybrid model
   python training/train_hybrid.py
   ```

## Documentation

- [Installation Guide](docs/installation.md)
- [Usage Instructions](docs/usage.md)
- [API Reference](docs/api_reference.md)

## License

This project is licensed under the MIT License.
