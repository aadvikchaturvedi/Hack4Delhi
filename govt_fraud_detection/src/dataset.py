import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

class GovSpendingDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        
        # Feature separation
        self.cont_cols = ['amount', 'days_since_last_contract', 'z_score_dept']
        self.cat_cols = ['dept_id', 'vendor_id', 'procurement_type']
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.cont_data = self.scaler.fit_transform(df[self.cont_cols]).astype('float32')
        
        self.label_encoders = []
        self.cat_data = []
        for col in self.cat_cols:
            le = LabelEncoder()
            self.cat_data.append(le.fit_transform(df[col]))
            self.label_encoders.append(le)
        self.cat_data = np.stack(self.cat_data, axis=1).astype('int64')
        
        self.labels = df['is_fraud'].values.astype('float32')

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return (torch.tensor(self.cont_data[idx]), 
                torch.tensor(self.cat_data[idx]), 
                torch.tensor(self.labels[idx]))