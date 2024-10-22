import torch
import torch.nn as nn
from transformers import AutoModelForQuestionAnswering, BertModel
import json
import numpy as np

model_name = 'bert-base-uncased'
model = BertModel.from_pretrained(model_name)
torch.save(model.state_dict(), 'bert-base-uncased.pth')

# model.reg_tokens = nn.Parameter(torch.randn(1, 768))
for k in model.state_dict().keys():
    print(k, model.state_dict()[k].shape)

# print(model.embeddings.word_embeddings.weight)
# print(typeccodel)
x = torch.randint(low=0, high=10, size=(1, 10))
out = model(x)
print(model.embeddings.word_embedding.grad)