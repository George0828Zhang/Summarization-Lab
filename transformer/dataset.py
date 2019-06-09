import torch
from torch.utils import data
import numpy as np
import json
from tqdm import tqdm_notebook as tqdm

class Dataset(data.Dataset):    
    def __init__(self, data_name, INPUT_MAX, OUTPUT_MAX, pad_idx, cutoff=None):
        print("loading json")
        data = json.load(open(data_name, 'r'))
        print("load json done.")
        sum_list = data['summary']
        data_list = data['document']
        
        if cutoff is not None:
            sum_list = sum_list[:cutoff]
            data_list = data_list[:cutoff]
        # idata -> list
        self.size = len(sum_list)
        self.sum_len = 0
        self.documents = []
        self.summaries = []
        
        for i in tqdm(range(len(sum_list))):
            if(len(data_list[i]) <= INPUT_MAX):
                data = data_list[i] + [pad_idx]*(INPUT_MAX-len(data_list[i]))
            else:
                data = data_list[i][:INPUT_MAX]
                
            if(len(sum_list[i]) <= OUTPUT_MAX):
                sum_in = sum_list[i] + [pad_idx]*(OUTPUT_MAX-len(sum_list[i]))
            else:
                sum_in = sum_list[i][:OUTPUT_MAX]
                
            self.documents.append(data)
            self.summaries.append(sum_in)
        
        # data_sz, 2, 
        self.documents = np.asarray(self.documents, dtype=np.int64)
        self.summaries = np.asarray(self.summaries, dtype=np.int64)
     
    def __len__(self):
        return self.size
    def __getitem__(self, index):
        return torch.from_numpy(self.documents[index]), torch.from_numpy(self.summaries[index])
    
def make_data_generator(data_name, in_max, out_max, padding_idx, batch_size, cutoff=None, shuffle=True, num_workers=4):    
    data_set = Dataset(data_name, in_max, out_max, padding_idx, cutoff)
    params = {'batch_size':batch_size,
         'shuffle': shuffle,
         'num_workers': num_workers}
    generator = data.DataLoader(data_set, **params)
    return data_set, generator