from torch.utils.data import Dataset
import os
from PIL import Image
import torch
class VQADataset(Dataset):
    def __init__(self, data, classes_to_idx = None, tokenize = None, max_seq_len=30, transform = None, root_dir = ''):
        self.transform = transform
        self.data = data
        self.max_seq_len = max_seq_len
        self.root_dir = root_dir
        self.classes_to_idx = classes_to_idx
        self.tokenize = tokenize
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        img_path = os.path.join(self.root_dir,self.data[index]['image_path'])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        question = self.data[index]['question']
        if self.tokenize:
            question = self.tokenize(question,self.max_seq_len)
            question = torch.tensor(question, dtype = torch.long)
        label = self.data[index]['answer']
        if self.classes_to_idx :
            label = self.classes_to_idx[label]
            label = torch.tensor(label, dtype = torch.long)
        return img, question, label