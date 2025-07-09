"""
Training script for Hybrid CNN-ViT models.
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.data_preprocessing import create_data_loaders
from models.hybrid_models import create_model


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} Training')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.3f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
        
        # Log to tensorboard
        if batch_idx % 50 == 0:
            step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), step)
            writer.add_scalar('Train/Accuracy', 100.*correct/total, step)
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device, epoch, writer):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} Validation')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.3f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    # Log to tensorboard
    writer.add_scalar('Val/Loss', epoch_loss, epoch)
    writer.add_scalar('Val/Accuracy', epoch_acc, epoch)
    
    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)


def main():
    parser = argparse.ArgumentParser(description='Train Hybrid CNN-ViT models for plant disease detection')
    parser.add_argument('--data_dir', type=str, default='data/plant_village/processed',
                        help='Path to processed dataset')
    parser.add_argument('--model', type=str, default='parallel_cnn_vit', 
                        choices=['hybrid_cnn_vit', 'parallel_cnn_vit', 'attention_fused_cnn_vit'],
                        help='Hybrid model to train')
    parser.add_argument('--cnn_backbone', type=str, default='resnet50',
                        choices=['resnet50', 'efficientnet_b0'],
                        help='CNN backbone for hybrid model')
    parser.add_argument('--vit_model', type=str, default='vit_base_patch16_224',
                        help='ViT model for hybrid architecture')
    parser.add_argument('--fusion_method', type=str, default='concat',
                        choices=['concat', 'add', 'multiply'],
                        help='Feature fusion method for parallel models')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--output_dir', type=str, default='results/model_weights',
                        help='Output directory for model weights')
    parser.add_argument('--log_dir', type=str, default='results/logs',
                        help='Directory for tensorboard logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create data loaders
    print('Loading data...')
    data_loaders = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size
    )
    
    # Get number of classes
    num_classes = len(data_loaders['train'].dataset.label_to_idx)
    print(f'Number of classes: {num_classes}')
    
    # Create model
    print(f'Creating {args.model} model...')
    model_kwargs = {
        'cnn_backbone': args.cnn_backbone,
        'vit_model': args.vit_model
    }
    
    if args.model == 'parallel_cnn_vit':
        model_kwargs['fusion_method'] = args.fusion_method
    
    model = create_model(args.model, num_classes, **model_kwargs)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Tensorboard writer
    model_name = f'{args.model}_{args.cnn_backbone}_{args.vit_model.replace("/", "_")}'
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, model_name))
    
    # Resume training if checkpoint provided
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume:
        print(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('accuracy', 0.0)
    
    # Training loop
    print('Starting training...')
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, data_loaders['train'], criterion, optimizer, device, epoch, writer
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(
            model, data_loaders['val'], criterion, device, epoch, writer
        )
        
        # Update learning rate
        scheduler.step()
        
        # Log epoch results
        print(f'Epoch {epoch}/{args.epochs-1}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_checkpoint_path = os.path.join(args.output_dir, f'best_{model_name}.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, best_checkpoint_path)
            print(f'  New best model saved with accuracy: {best_acc:.2f}%')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'{model_name}_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, checkpoint_path)
    
    # Save final model
    final_checkpoint_path = os.path.join(args.output_dir, f'final_{model_name}.pth')
    save_checkpoint(model, optimizer, args.epochs-1, val_loss, val_acc, final_checkpoint_path)
    
    # Save training configuration
    config = {
        'model': args.model,
        'cnn_backbone': args.cnn_backbone,
        'vit_model': args.vit_model,
        'fusion_method': args.fusion_method if args.model == 'parallel_cnn_vit' else None,
        'num_classes': num_classes,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'image_size': args.image_size,
        'best_accuracy': best_acc,
        'label_to_idx': data_loaders['train'].dataset.label_to_idx
    }
    
    config_path = os.path.join(args.output_dir, f'config_{model_name}.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    writer.close()
    print(f'Training completed! Best validation accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()
