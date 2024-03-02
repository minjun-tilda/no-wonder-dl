import torch
import math
from sklearn.metrics import mean_squared_error

def validate(model, dataloader, subset, pad_idx = None):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            predictions = model((), X_batch, pad_idx)
            # 스케일링된 값을 원래 크기로 되돌림
            predictions_scaled = subset.dataset.target_scaler.inverse_transform(predictions.numpy().reshape(-1, 1))
            y_batch_scaled = subset.dataset.target_scaler.inverse_transform(y_batch.numpy().reshape(-1, 1))
            
            # RMSE 계산
            mse = mean_squared_error(y_batch_scaled, predictions_scaled)
            rmse = math.sqrt(mse)
            total_loss += rmse

    avg_loss = total_loss / len(dataloader)
    return avg_loss