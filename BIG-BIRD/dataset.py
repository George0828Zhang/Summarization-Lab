import torch
from torch.utils import data
import numpy as np
import json
from tqdm import tqdm_notebook as tqdm
import random
#from tqdm import tqdm

class Dataset(data.Dataset):    
    def __init__(self, data_name, INPUT_MAX, OUTPUT_MAX, pad_idx, cutoff=None):
        print("loading json")
        data = json.load(open(data_name, 'r'))
        print("load json done.")
        sum_list = data['summary']
        data_list = data['text']
        self.scores = np.asarray(data['score'], dtype=np.float64)
        
        if cutoff is not None:
            sum_list = sum_list[:cutoff]
            data_list = data_list[:cutoff]
        # idata -> list
        self.size = len(sum_list)
            
        self.src = []
        self.tgt = []
        
        self.pad_idx = pad_idx
        
        for i in tqdm(range(self.size)):
            src = data_list[i][:INPUT_MAX]
            tgt = sum_list[i][:OUTPUT_MAX]
            self.src.append(src)
            self.tgt.append(tgt)
                    
        idx = np.argsort([len(x) for x in self.src])[::-1] # descending
        
        self.src = [ self.src[i] for i in idx]
        self.tgt = [ self.tgt[i] for i in idx]
        
    def np_jagged(self, array):
        MAX = max([len(i) for i in array])
        out = [ a + [self.pad_idx]*(MAX-len(a)) if len(a) < MAX else a[:MAX] for a in array ]
        return np.asarray(out, dtype=np.int64)
            
     
    def make_generator(self, batch_size, shuffle=True):
        # preprocess
        total = self.size // batch_size
        if total * batch_size < self.size:
            total += 1
        
        iters = np.random.randint(low=0,high=total,size=total) if shuffle else range(total)
        for i in iters:            
            fr = i*batch_size
            to = min(fr+batch_size, self.size)
            src = self.np_jagged(self.src[fr:to])
            tgt = self.np_jagged(self.tgt[fr:to])
            yield torch.from_numpy(src), torch.from_numpy(tgt)
            

class PretrainDataset(data.Dataset):    
    def __init__(self, data_name, INPUT_MAX, OUTPUT_MAX, pad_idx, cutoff=None):
        print("loading json")
        data = json.load(open(data_name, 'r'))
        print("load json done.")
        data_list = data['text']
        
        if cutoff is not None:
            data_list = data_list[:cutoff]
        # idata -> list
        self.size = len(data_list)
                
        self.src = []
        self.tgt = []
        
        self.pad_idx = pad_idx
        
        for i in tqdm(range(self.size)):
            src = list(data_list[i])[:INPUT_MAX]
            tgt = data_list[i][:OUTPUT_MAX]
            random.shuffle(src)
            self.src.append(src)
            self.tgt.append(tgt)
                    
        idx = np.argsort([len(x) for x in self.tgt])[::-1] # descending
        
        self.src = [ self.src[i] for i in idx]
        self.tgt = [ self.tgt[i] for i in idx]
        
    def np_jagged(self, array):
        MAX = max([len(i) for i in array])
        out = [ a + [self.pad_idx]*(MAX-len(a)) if len(a) < MAX else a[:MAX] for a in array ]
        return np.asarray(out, dtype=np.int64)
            
     
    def make_generator(self, batch_size, shuffle=True):
        # preprocess
        total = self.size // batch_size
        if total * batch_size < self.size:
            total += 1
        
        iters = np.random.randint(low=0,high=total,size=total) if shuffle else range(total)
        for i in iters:            
            fr = i*batch_size
            to = min(fr+batch_size, self.size)
            src = self.np_jagged(self.src[fr:to])
            tgt = self.np_jagged(self.tgt[fr:to])
            yield torch.from_numpy(src), torch.from_numpy(tgt)
        
    
    
def make_data_generator(data_name, in_max, out_max, padding_idx, batch_size, pretrain=False, cutoff=None, shuffle=True, num_workers=4):    
    if pretrain:
        data_set = PretrainDataset(data_name, in_max, out_max, padding_idx, cutoff)
    else:
        data_set = Dataset(data_name, in_max, out_max, padding_idx, cutoff)
    return data_set, data_set.make_generator(batch_size=batch_size, shuffle=shuffle)
    #params = {'batch_size':batch_size,
    #     'shuffle': shuffle,
    #     'num_workers': num_workers}
    #generator = data.DataLoader(data_set, **params)
    #return data_set, generator