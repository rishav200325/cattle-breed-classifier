import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

import torch
import argparse
from ml.src.utils.io import load_config
from ml.src.data.transforms import get_train_transforms, get_eval_transforms
from ml.src.utils.manifests import create_dataloaders
from ml.src.models.resnet import CattleResNet
from ml.src.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Local ResNet training runner')
    parser.add_argument('--epochs', type=int, help='Total number of training epochs')
    parser.add_argument('--phase1-epochs', type=int, default=None, help='Epochs for frozen-head training phase')
    parser.add_argument('--batch-size', type=int, help='Batch size for training')
    parser.add_argument('--num-workers', type=int, help='Number of dataloader workers')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config('resnet', root / 'ml' / 'configs')
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['num_epochs'] = args.epochs

    print('Using config:')
    print(config['model'])
    print(config['training'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    if not torch.cuda.is_available() and args.epochs is None:
        config['training']['num_epochs'] = min(config['training'].get('num_epochs', 30), 10)
        print('No GPU detected: limiting total epochs to', config['training']['num_epochs'])

    train_transform = get_train_transforms(
        img_size=config['image']['size'],
        random_crop=config['augmentation']['random_crop'],
        horizontal_flip=config['augmentation']['horizontal_flip'],
        rotation_degrees=config['augmentation']['rotation_degrees'],
        color_jitter=config['augmentation']['color_jitter'],
        color_jitter_strength=config['augmentation']['color_jitter_strength'],
    )
    eval_transform = get_eval_transforms(img_size=config['image']['size'])

    dataloaders = create_dataloaders(
        manifests_dir=root / config['data']['manifests_dir'],
        data_root=root / config['data']['raw_dir'],
        train_transform=train_transform,
        eval_transform=eval_transform,
        batch_size=config['training']['batch_size'],
        num_workers=args.num_workers or min(4, os.cpu_count() or 1),
    )

    print('Datasets:')
    for split in ['train', 'val', 'test']:
        ds = dataloaders.get(split)
        if ds is None:
            print(f'  {split}: MISSING')
        else:
            print(f'  {split}: {len(ds.dataset)} images, {ds.dataset.num_classes} classes')

    class_names = dataloaders['train'].dataset.classes
    model = CattleResNet(
        num_classes=len(class_names),
        backbone=config['model']['architecture']['backbone'],
        pretrained=config['model']['architecture']['pretrained'],
        freeze_backbone=config['model']['architecture']['freeze_backbone'],
    )

    trainer = Trainer(model, config, dataloaders, class_names, device, model_name='resnet')

    phase1_epochs = args.phase1_epochs if args.phase1_epochs is not None else config['model']['architecture'].get('unfreeze_after_epochs', 10)
    phase2_lr = config['training'].get('fine_tune_lr', 1e-4)
    unfreeze_layers = config['model']['architecture'].get('unfreeze_layers', None)

    print('Starting phase-switch training:')
    print('Phase 1 epochs:', phase1_epochs)
    summary = trainer.train_with_phase_switch(
        phase1_epochs=phase1_epochs,
        phase2_lr=phase2_lr,
        unfreeze_layers=unfreeze_layers,
    )
    print('Training summary:', summary)

    print('Evaluating on test set...')
    test_results = trainer.evaluate('test')
    print('Test results:', test_results)

    checkpoint = {
        'model_name': 'resnet',
        'model_state_dict': trainer.model.state_dict(),
        'class_names': class_names,
    }

    checkpoint_path = root / 'ml' / 'artifacts' / 'checkpoints' / 'resnet_best.pth'
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print('Saved checkpoint to', checkpoint_path)

    legacy_path = root / 'models' / 'cattle_breed_classifier_full_model.pth'
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, legacy_path)
    print('Saved legacy checkpoint to', legacy_path)

    classes_file = root / 'models' / 'classes.txt'
    with open(classes_file, 'w', encoding='utf-8') as f:
        f.write(','.join(class_names))
    print('Saved classes to', classes_file)


if __name__ == '__main__':
    main()
