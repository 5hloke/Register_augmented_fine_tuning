import torch
from transformers import BertModel
import numpy as np

model_name = 'bert-base-uncased'
model = BertModel.from_pretrained(model_name)
torch.save(model.state_dict(), 'bert-base-uncased.pth')
