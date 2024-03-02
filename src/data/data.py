import yaml
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, config, idx=None):
        self.data_frame = pd.read_csv(config['data_path'])
        self.numerical_features = config['numerical_feature']

        if idx is not None:
            self.numerical_features.remove(config['numerical_feature'][idx])

        print(f'numerical_type 확인: {self.numerical_features}')
        self.target_feature = config['target_feature']

        # 데이터 전처리
        self.X = self.data_frame[self.numerical_features].values
        self.y = self.data_frame[self.target_feature].values.reshape(-1, 1)

        # StandardScaler 적용
        self.feature_scaler = StandardScaler()
        self.X = self.feature_scaler.fit_transform(self.X)

        self.target_scaler = StandardScaler()
        self.y = self.target_scaler.fit_transform(self.y)
        
        self.y = torch.tensor(self.y, dtype=torch.float32).flatten()

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        X = torch.tensor(X, dtype=torch.float32)
        return X, y

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    return config

def create_data_loaders(config, idx = None, batch_size=512, split_ratio=0.8):
    dataset = CustomDataset(config=config, idx=idx)
    
    # 데이터셋 크기 계산 및 분할
    train_size = int(len(dataset) * split_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # 훈련 및 테스트 DataLoader 생성
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader, train_dataset, test_dataset