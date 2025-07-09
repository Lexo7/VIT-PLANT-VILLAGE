"""
Model evaluation script for plant disease detection.
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.data_preprocessing import create_data_loaders
from models.cnn_models import ResNetClassifier, EfficientNetClassifier
from models.vit_models import VisionTransformer, CustomViT
from models.hybrid_models import create_model


def load_model(model_type, checkpoint_path, num_classes, device, **kwargs):
    """Load a trained model from checkpoint."""
    
    if model_type == 'resnet50':
        model = ResNetClassifier(num_classes=num_classes)
    elif model_type == 'efficientnet_b0':
        model = EfficientNetClassifier(num_classes=num_classes)
    elif model_type == 'custom_vit':
        model = CustomViT(num_classes=num_classes, **kwargs)
    elif model_type.startswith('vit_'):
        model = VisionTransformer(num_classes=num_classes, model_name=model_type)
    elif model_type in ['hybrid_cnn_vit', 'parallel_cnn_vit', 'attention_fused_cnn_vit']:
        model = create_model(model_type, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_model(model, dataloader, device, class_names):
    """Evaluate model on a dataset."""
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate accuracy
    accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
    
    # Generate classification report
    report = classification_report(
        all_targets, all_predictions, 
        target_names=class_names, 
        output_dict=True
    )
    
    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': np.array(all_probabilities),
        'classification_report': report,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot confusion matrix."""
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def save_results(results, output_dir, model_name):
    """Save evaluation results."""
    
    # Save classification report
    report_df = pd.DataFrame(results['classification_report']).transpose()
    report_path = os.path.join(output_dir, f'{model_name}_classification_report.csv')
    report_df.to_csv(report_path)
    
    # Save confusion matrix
    cm_path = os.path.join(output_dir, f'{model_name}_confusion_matrix.npy')
    np.save(cm_path, results['confusion_matrix'])
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'true_label': results['targets'],
        'predicted_label': results['predictions'],
    })
    predictions_path = os.path.join(output_dir, f'{model_name}_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    
    # Save probabilities
    prob_path = os.path.join(output_dir, f'{model_name}_probabilities.npy')
    np.save(prob_path, results['probabilities'])
    
    # Save summary
    summary = {
        'accuracy': float(results['accuracy']),
        'macro_avg_f1': results['classification_report']['macro avg']['f1-score'],
        'weighted_avg_f1': results['classification_report']['weighted avg']['f1-score'],
    }
    
    summary_path = os.path.join(output_dir, f'{model_name}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to model configuration file')
    parser.add_argument('--data_dir', type=str, default='data/plant_village/processed',
                        help='Path to processed dataset')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--output_dir', type=str, default='results/evaluation_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create data loaders
    data_loaders = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=config['image_size']
    )
    
    # Get class names
    idx_to_label = {v: k for k, v in config['label_to_idx'].items()}
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    
    # Load model
    print(f"Loading model: {config['model']}")
    model_kwargs = {}
    
    if 'cnn_backbone' in config:
        model_kwargs['cnn_backbone'] = config['cnn_backbone']
    if 'vit_model' in config:
        model_kwargs['vit_model'] = config['vit_model']
    if 'fusion_method' in config and config['fusion_method']:
        model_kwargs['fusion_method'] = config['fusion_method']
    
    model = load_model(
        config['model'], 
        args.checkpoint, 
        config['num_classes'], 
        device,
        **model_kwargs
    )
    
    # Evaluate model
    print(f"Evaluating on {args.split} set...")
    results = evaluate_model(model, data_loaders[args.split], device, class_names)
    
    # Print results
    print(f"\nResults on {args.split} set:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro F1: {results['classification_report']['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1: {results['classification_report']['weighted avg']['f1-score']:.4f}")
    
    # Plot confusion matrix
    model_name = Path(args.checkpoint).stem
    cm_path = os.path.join(args.output_dir, f'{model_name}_confusion_matrix.png')
    plot_confusion_matrix(results['confusion_matrix'], class_names, cm_path)
    
    # Save results
    summary = save_results(results, args.output_dir, model_name)
    
    print("\nClassification Report:")
    print(pd.DataFrame(results['classification_report']).transpose())


if __name__ == '__main__':
    main()
