from src.model import TransformerFraudModel
from src.dataset import GovSpendingDataset
from src.trainer import train_model
from torch.utils.data import DataLoader

def main():
    # 1. Load Data
    dataset = GovSpendingDataset('data/raw_spending.csv')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 2. Define Model Specs
    # cat_cardinalities is the number of unique vendors, depts, etc.
    cat_cards = [dataset.cat_data[:, i].max() + 1 for i in range(len(dataset.cat_cols))]
    
    model = TransformerFraudModel(
        n_cont=len(dataset.cont_cols),
        cat_cardinalities=cat_cards
    )
    
    # 3. Train
    print("Starting Transformer Training for Gov Oversight...")
    train_model(model, loader, epochs=20)
    
    # 4. Save for Deployment
    torch.save(model.state_dict(), 'models/transformer_fraud_v1.pth')
    print("Model saved to models/ directory.")

if __name__ == "__main__":
    main()