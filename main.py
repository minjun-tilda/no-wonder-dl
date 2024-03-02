import argparse

from src.model.fttransformer import FTTransformer
from src.data.data import load_yaml, create_data_loaders

from src.train.train import train
from src.train.valid import validate

import torch
from torch import nn
import torch.optim as optim

import numpy as np
import random
import os
import copy
from tqdm import tqdm

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # The following two lines are for ensuring deterministic behavior in PyTorch
    # but might reduce performance and are not suitable for all models.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(42)

    parser = argparse.ArgumentParser(description='Process a YAML config file.')
    parser.add_argument('config', type=str, help='Path to the YAML config file')
    
    args = parser.parse_args()
    
    config = load_yaml(args.config)

    train_dataloader, test_dataloader, train_dataset, test_dataset = create_data_loaders(copy.deepcopy(config))

    model = FTTransformer(
        sh_categorical=(),
        n_numerical=21,
        n_layers=3,
        n_head=8,
        d_k=16,
        d_v=16,
        d_model=196,
        d_inner=784,
        d_out=1
    )

    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    patience = 10
    epochs_since_improvement = 0

    epochs = 100
    for epoch in tqdm(range(epochs)):
        train(model, train_dataloader, loss_fn, optimizer)
        val_loss = validate(model, test_dataloader, test_dataset)

        # Check if the validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            # Save the model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_since_improvement += 1

        # Early stopping
        if epochs_since_improvement >= patience:
            print(f"Stopping early due to no improvement in validation loss. Epoch {epoch+1}, Validation Loss: {best_val_loss}")
            break
    
    print('원본 모델 확인 끝!')
    model.load_state_dict(torch.load('best_model.pth'))

    for idx in range(21):
        val_loss = validate(model, test_dataloader, test_dataset, idx)
        print(f"IDX: {idx} | Validation Loss: {val_loss}")

    for idx in range(21):
        print(f'{idx}에 대해 실제 성능 확인 진행!')
        train_dataloader, test_dataloader, train_dataset, test_dataset = create_data_loaders(copy.deepcopy(config), idx)

        model = FTTransformer(
            sh_categorical=(),
            n_numerical=20, ## 하나 줄임
            n_layers=3,
            n_head=8,
            d_k=16,
            d_v=16,
            d_model=196,
            d_inner=784,
            d_out=1
        )

        loss_fn = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001)

        best_val_loss = float('inf')
        patience = 10
        epochs_since_improvement = 0

        epochs = 100
        for epoch in tqdm(range(epochs)):
            train(model, train_dataloader, loss_fn, optimizer)
            val_loss = validate(model, test_dataloader, test_dataset)

            # Check if the validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            # Early stopping
            if epochs_since_improvement >= patience:
                print(f"Stopping early due to no improvement in validation loss. Epoch {epoch+1}, Validation Loss: {best_val_loss}")
                break


if __name__ == '__main__':
    main()