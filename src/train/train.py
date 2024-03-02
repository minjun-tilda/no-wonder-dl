def train(model, dataloader, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        # Forward pass
        predictions = model((), X_batch)
        loss = loss_fn(predictions, y_batch.unsqueeze(1))
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss