import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image

class CUB200Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.num_classes = self.data['label'].nunique()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.data.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_data(csv_file, img_dir, batch_size=32, transform=None):
    dataset = CUB200Dataset(csv_file, img_dir, transform=transform)
    
    train_indices = dataset.data[dataset.data['is_train'] == 1].index.tolist()
    test_indices = dataset.data[dataset.data['is_train'] == 0].index.tolist()
    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
