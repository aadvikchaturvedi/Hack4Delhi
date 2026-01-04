import torch.optim as optim

def train_model(model, dataloader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for cont, cat, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(cont, cat).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f}")

def get_anomaly_score(model, cont, cat):
    """
    Uses the attention weights to flag transactions that deviate 
    from learned government spending patterns.
    """
    model.eval()
    with torch.no_grad():
        score = model(cont, cat)
    return score