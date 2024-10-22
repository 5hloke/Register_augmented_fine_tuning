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

'''
BertModel - > RegBert
BertLayer - > Block 
BertEncoder -> RegBertEncoder

'''
class RegBert(BertModel):
    def __init__(self, config, num_registers=10, device='mps'):
        super(RegBert, self).__init__(config)
        self.config = config
        self.reg_tokens = nn.Parameter(torch.zeros(num_registers, 768))
        self.reg_pos = nn.Parameter(torch.zeros(num_registers, 768))

    def forward(self, x):
        pass
    
model = RegBert.from_pretrained('bert-base-uncased')
print(model.config)
for name in model.state_dict().keys():
    print(name)

# Block
# Feed Forward