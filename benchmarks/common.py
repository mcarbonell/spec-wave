import os
import sys
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Fix Windows console encoding for UTF-8 output
try:
    if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    try:
        import torch_directml
        return torch_directml.device()
    except Exception:
        return torch.device('cpu')

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class TokenBlockDataset(Dataset):
    """
    Packs raw text tokens into fixed-length contiguous blocks of seq_len tokens.
    """
    def __init__(self, token_list, seq_len=64, max_blocks=None, stride=None):
        if stride is None:
            stride = seq_len
        blocks = []
        for i in range(0, len(token_list) - seq_len + 1, stride):
            blocks.append(token_list[i : i + seq_len])
            if max_blocks is not None and len(blocks) >= max_blocks:
                break
        self.blocks = torch.tensor(blocks, dtype=torch.long)
        
    def __len__(self):
        return len(self.blocks)
        
    def __getitem__(self, idx):
        return self.blocks[idx]

def load_wikitext2_tokens(tokenizer=None):
    """
    Loads WikiText-2 real train and test splits, returns lists of token ids.
    Uses local cache in data/ directory for instantaneous loading.
    """
    import urllib.request
    import tiktoken
    
    os.makedirs('data', exist_ok=True)
    train_path = os.path.join('data', 'wikitext2_train.txt')
    test_path = os.path.join('data', 'wikitext2_test.txt')
    
    url_train = 'https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt'
    url_test = 'https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/test.txt'
    
    if not os.path.exists(train_path):
        print('Downloading WikiText-2 train.txt...')
        urllib.request.urlretrieve(url_train, train_path)
    if not os.path.exists(test_path):
        print('Downloading WikiText-2 test.txt...')
        urllib.request.urlretrieve(url_test, test_path)
        
    with open(train_path, 'r', encoding='utf-8') as f:
        train_text = f.read()
    with open(test_path, 'r', encoding='utf-8') as f:
        test_text = f.read()
        
    if tokenizer is None:
        enc = tiktoken.get_encoding('gpt2')
        train_tokens = enc.encode(train_text)
        test_tokens = enc.encode(test_text)
    else:
        train_tokens = tokenizer.encode(train_text)
        test_tokens = tokenizer.encode(test_text)
        
    return train_tokens, test_tokens
