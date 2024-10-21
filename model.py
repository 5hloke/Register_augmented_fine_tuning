import torch
import numpy as np 
import torch.nn as nn
from transformers import BertModel, AutoModelForQuestionAnswering

# Creatinf the Encoder Only model to read the weights from bert_base_encased.pth 
'''
Example of a single Layer of Bert Model:
encoder.layer.0.attention.self.query.weight torch.Size([768, 768])
encoder.layer.0.attention.self.query.bias torch.Size([768])
encoder.layer.0.attention.self.key.weight torch.Size([768, 768])
encoder.layer.0.attention.self.key.bias torch.Size([768])
encoder.layer.0.attention.self.value.weight torch.Size([768, 768])
encoder.layer.0.attention.self.value.bias torch.Size([768])
encoder.layer.0.attention.output.dense.weight torch.Size([768, 768])
encoder.layer.0.attention.output.dense.bias torch.Size([768])
encoder.layer.0.attention.output.LayerNorm.weight torch.Size([768])
encoder.layer.0.attention.output.LayerNorm.bias torch.Size([768])
encoder.layer.0.intermediate.dense.weight torch.Size([3072, 768])
encoder.layer.0.intermediate.dense.bias torch.Size([3072])
encoder.layer.0.output.dense.weight torch.Size([768, 3072])
encoder.layer.0.output.dense.bias torch.Size([768])
encoder.layer.0.output.LayerNorm.weight torch.Size([768])
encoder.layer.0.output.LayerNorm.bias torch.Size([768])
'''
# Attention

class RegBert(torch.nn.Module):
    
    def __init__(self, num_register = 10, model_path ='bert-base-uncased.pth', device = 'mps'): 
        super(RegBert, self).__init__()
        self.num_register = num_register
        self.reg_tokens = nn.Parameter(torch.zeros(self.num_register, 768)).to(device)
        self.reg_pos = nn.Parameter(torch.zeros(self.num_register, 768)).to(device)
        model_name = 'bert-base-uncased'
        self.encoder = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
        self.device = device
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        pass

reg_bert = RegBert()
        
    
# Block
# Feed Forward