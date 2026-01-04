import torch
import os
from torch.utils.data import DataLoader

# Modular imports from your src directory
from src.ingestion import IndiaDataIngestor
from src.labeling import apply_indian_red_flags
from src.dataset import GovSpendingDataset
from src.model import TransformerFraudModel
from src.trainer import train_model

def main():
    # Configuration
    API_KEY = os.getenv("OGD_INDIA_API_KEY", "YOUR_DEFAULT_KEY")
    RESOURCE_ID = "f507b5e3-9977-463d-..." # Replace with actual Assam/GeM ID
    DATA_PATH = 'data/processed_india_spending.csv'
    MODEL_SAVE_PATH = 'models/transformer_fraud_v1.pth'

    # 1. Ingest Data
    print("--- Phase 1: Data Ingestion ---")
    try:
        ingestor = IndiaDataIngestor(api_key=API_KEY)
        raw_df = ingestor.fetch_resource(RESOURCE_ID, limit=5000)
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return

    # 2. Process & Label (Indian Context)
    print("--- Phase 2: Heuristic Labeling ---")
    labeled_df = apply_indian_red_flags(raw_df)
    labeled_df.to_csv(DATA_PATH, index=False)
    
    # 3. Prepare Pipeline
    print("--- Phase 3: Dataset Preparation ---")
    dataset = GovSpendingDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Calculate cardinalities for the Transformer's embedding layers
    cat_cards = [dataset.cat_data[:, i].max() + 1 for i in range(len(dataset.cat_cols))]
    
    model = TransformerFraudModel(
        n_cont=len(dataset.cont_cols),
        cat_cardinalities=cat_cards,
        embed_dim=64 # Contextual embedding size
    )
    
    # 4. Train
    print("--- Phase 4: Transformer Training ---")
    # You can move device logic (CPU/GPU) here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_model(model, loader, epochs=20)
    
    # 5. Save Model
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Deployment-ready model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()