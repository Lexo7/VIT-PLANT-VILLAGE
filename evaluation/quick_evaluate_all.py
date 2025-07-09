#!/usr/bin/env python3
"""
Quick evaluation script for all trained models.
This script automatically finds and evaluates all trained models.
"""

import os
import sys
import json
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.evaluate_models import load_model, evaluate_model, plot_confusion_matrix, save_results
from data.data_preprocessing import create_data_loaders
import torch

def find_model_checkpoints(weights_dir="results/model_weights"):
    """Find all model checkpoints."""
    checkpoints = []
    
    # Look for checkpoint files
    for checkpoint_file in glob.glob(os.path.join(weights_dir, "*.pth")):
        model_name = os.path.basename(checkpoint_file).replace('.pth', '')
        checkpoints.append({
            'name': model_name,
            'path': checkpoint_file
        })
    
    return checkpoints

def infer_model_config(model_name):
    """Infer model configuration from model name."""
    config = {
        'image_size': 224,
        'model': model_name
    }
    
    # CNN models
    if 'resnet50' in model_name:
        config['model'] = 'resnet50'
    elif 'efficientnet_b0' in model_name:
        config['model'] = 'efficientnet_b0'
    
    # ViT models
    elif 'vit_base_patch16_224' in model_name:
        config['model'] = 'vit_base_patch16_224'
    elif 'vit_small_patch16_224' in model_name:
        config['model'] = 'vit_small_patch16_224'
    elif 'custom_vit' in model_name:
        config['model'] = 'custom_vit'
    
    # Hybrid models
    elif 'hybrid_cnn_vit' in model_name:
        config['model'] = 'hybrid_cnn_vit'
        config['cnn_backbone'] = 'resnet50'
        config['vit_model'] = 'vit_base_patch16_224'
    elif 'parallel_cnn_vit' in model_name:
        config['model'] = 'parallel_cnn_vit'
        config['cnn_backbone'] = 'resnet50'
        config['vit_model'] = 'vit_base_patch16_224'
        config['fusion_method'] = 'concat'
    elif 'attention_fused_cnn_vit' in model_name:
        config['model'] = 'attention_fused_cnn_vit'
        config['cnn_backbone'] = 'resnet50'
        config['vit_model'] = 'vit_base_patch16_224'
    
    return config

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create data loaders
    print('Loading data...')
    data_loaders = create_data_loaders(
        data_dir='data/plant_village/processed',
        batch_size=32,
        num_workers=4,
        image_size=224
    )
    
    # Get class information
    class_names = list(data_loaders['train'].dataset.label_to_idx.keys())
    num_classes = len(class_names)
    
    # Find all model checkpoints
    checkpoints = find_model_checkpoints()
    
    if not checkpoints:
        print("❌ No model checkpoints found in results/model_weights/")
        print("Please train models first using the training scripts.")
        return
    
    print(f"Found {len(checkpoints)} model checkpoints:")
    for cp in checkpoints:
        print(f"  - {cp['name']}")
    
    # Create output directory
    output_dir = 'results/evaluation'
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate each model
    all_results = []
    
    for checkpoint in checkpoints:
        print(f"\n🔍 Evaluating {checkpoint['name']}...")
        
        try:
            # Infer model configuration
            config = infer_model_config(checkpoint['name'])
            
            # Load model
            model_kwargs = {}
            if 'cnn_backbone' in config:
                model_kwargs['cnn_backbone'] = config['cnn_backbone']
            if 'vit_model' in config:
                model_kwargs['vit_model'] = config['vit_model']
            if 'fusion_method' in config:
                model_kwargs['fusion_method'] = config['fusion_method']
            
            model = load_model(
                model_type=config['model'],
                checkpoint_path=checkpoint['path'],
                num_classes=num_classes,
                device=device,
                **model_kwargs
            )
            
            # Evaluate on test set
            results = evaluate_model(
                model=model,
                dataloader=data_loaders['test'],
                device=device,
                class_names=class_names
            )
            
            # Save results
            summary = save_results(results, output_dir, checkpoint['name'])
            summary['model_name'] = checkpoint['name']
            all_results.append(summary)
            
            # Plot confusion matrix
            plot_confusion_matrix(
                results['confusion_matrix'],
                class_names,
                save_path=os.path.join(output_dir, f"{checkpoint['name']}_confusion_matrix.png")
            )
            
            print(f"✅ {checkpoint['name']}: Accuracy = {results['accuracy']:.4f}")
            
        except Exception as e:
            print(f"❌ Error evaluating {checkpoint['name']}: {str(e)}")
            continue
    
    # Generate comparison report
    if all_results:
        print("\n📊 Generating comparison report...")
        
        # Create results DataFrame
        df_results = pd.DataFrame(all_results)
        df_results = df_results.sort_values('accuracy', ascending=False)
        
        # Save comparison CSV
        df_results.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
        
        # Create visualization
        plt.figure(figsize=(14, 10))
        
        # Accuracy comparison
        plt.subplot(2, 2, 1)
        sns.barplot(data=df_results, x='accuracy', y='model_name', palette='viridis')
        plt.title('Model Accuracy Comparison')
        plt.xlabel('Accuracy')
        plt.ylabel('Model')
        
        # F1 Score comparison
        plt.subplot(2, 2, 2)
        sns.barplot(data=df_results, x='macro_avg_f1', y='model_name', palette='plasma')
        plt.title('Macro F1 Score Comparison')
        plt.xlabel('Macro F1 Score')
        plt.ylabel('Model')
        
        # Weighted F1 Score comparison
        plt.subplot(2, 2, 3)
        sns.barplot(data=df_results, x='weighted_avg_f1', y='model_name', palette='coolwarm')
        plt.title('Weighted F1 Score Comparison')
        plt.xlabel('Weighted F1 Score')
        plt.ylabel('Model')
        
        # Model type analysis
        plt.subplot(2, 2, 4)
        model_types = []
        for model in df_results['model_name']:
            if 'resnet' in model or 'efficientnet' in model:
                model_types.append('CNN')
            elif 'vit' in model and 'hybrid' not in model and 'parallel' not in model:
                model_types.append('ViT')
            else:
                model_types.append('Hybrid')
        
        df_results['model_type'] = model_types
        type_performance = df_results.groupby('model_type')['accuracy'].mean()
        sns.barplot(x=type_performance.index, y=type_performance.values, palette='Set2')
        plt.title('Average Accuracy by Model Type')
        plt.xlabel('Model Type')
        plt.ylabel('Average Accuracy')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Generate text report
        report = f"""# Plant Disease Detection - Evaluation Results

## Model Performance Summary

### Top 3 Performing Models:
"""
        
        for i, (_, row) in enumerate(df_results.head(3).iterrows()):
            report += f"""
{i+1}. **{row['model_name']}**
   - Accuracy: {row['accuracy']:.4f}
   - Macro F1: {row['macro_avg_f1']:.4f}
   - Weighted F1: {row['weighted_avg_f1']:.4f}
"""
        
        report += f"""

### Model Type Analysis:
"""
        
        for model_type, avg_acc in type_performance.items():
            report += f"\n- {model_type}: {avg_acc:.4f} average accuracy"
        
        report += f"""

## Conclusions

Best performing model: **{df_results.iloc[0]['model_name']}** with {df_results.iloc[0]['accuracy']:.4f} accuracy

The results show that {type_performance.idxmax()} models perform best on this plant disease detection task.

## Files Generated
- Individual model results: `results/evaluation/[model_name]_*`
- Comparison visualization: `results/evaluation/model_comparison.png`
- Comparison data: `results/evaluation/model_comparison.csv`
- This report: `results/evaluation/evaluation_report.md`
"""
        
        # Save report
        with open(os.path.join(output_dir, 'evaluation_report.md'), 'w') as f:
            f.write(report)
        
        print(report)
        print("\n✅ Evaluation complete!")
        print(f"📁 Results saved to: {output_dir}")
        
    else:
        print("\n❌ No models were successfully evaluated.")

if __name__ == "__main__":
    main()
