import torch
import torch.nn as nn

class TransformerFraudModel(nn.Module):
    def __init__(self, n_cont, cat_cardinalities, embed_dim=64, n_heads=4, n_layers=3):
        super().__init__()
        # Continuous feature projection
        self.cont_proj = nn.Linear(n_cont, n_cont * embed_dim)
        
        # Categorical embeddings
        self.cat_embeds = nn.ModuleList([
            nn.Embedding(card + 1, embed_dim) for card in cat_cardinalities
        ])
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output Heads
        self.fraud_classifier = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_cont, x_cat):
        # 1. Embed categorical data
        cat_embeddings = [embed(x_cat[:, i]) for i, embed in enumerate(self.cat_embeds)]
        cat_embeddings = torch.stack(cat_embeddings, dim=1) # [Batch, Num_Cat, Embed_Dim]
        
        # 2. Project continuous data
        cont_embeddings = self.cont_proj(x_cont).view(x_cont.size(0), -1, cat_embeddings.size(-1))
        
        # 3. Combine and pass through Transformer
        x = torch.cat([cat_embeddings, cont_embeddings], dim=1)
        x = self.transformer(x)
        
        # 4. Pooling: Take the mean of the sequence for classification
        x_avg = x.mean(dim=1)
        return self.fraud_classifier(x_avg)