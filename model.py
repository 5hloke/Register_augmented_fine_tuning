import torch
import numpy as np 
import torch.nn as nn
from transformers import BertModel, AutoModelForQuestionAnswering, BertLayer

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

class Encoder(nn.Module):
    def __init__(self, config): 
        super().__init__()
        self.config =config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False 
    
    def forward(self): 
        pass
class RegBert(BertModel):
    def __init__(self, config, num_registers=10, device='mps'):
        super().__init__(config)
        self.config = config
        self.reg_tokens = nn.Parameter(torch.zeros(1, num_registers, 768))
        self.reg_pos = nn.Parameter(torch.zeros(1, num_registers, 768))
        self.device = device
        self.encoder = Encoder(self.config)
        self.add = Add()
        trunc_normal_(self.reg_tokens, std=.02)
        trunc_normal_(self.reg_pos, std=.02)


    def forward(self, input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None):
        
        
        batch_size, seq_length = input_ids.size()
        input_shape = input_ids.size()
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.device)
        
        # Here are the positional embeddings + word embeddings + token type embeddings
        input_embeds = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)

        register = self.reg_tokens.expand(batch_size, -1, -1)
        register = self.add(register, self.reg_pos)
        embedding_output = torch.cat((register, input_embeds), dim=1)

        # use_sdpa_attention_masks = (
        #     self.attn_implementation == "sdpa"
        #     and self.position_embedding_type == "absolute"
        #     and head_mask is None
        #     and not output_attentions
        # ) 

        #prepare 4D attention mask if needed
        def prepare_mask(mask, dtype, target_length):
            batch, key_length = mask.shape 
            target_length = target_length if target_length is not None else key_length
            expanded_mask = mask[:, None, None, :].expand(batch, 1, target_length, key_length).to(self.device)
            inverted_mask = 1.0 - expanded_mask
            return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)
        extended_attention_mask = prepare_mask(attention_mask, embedding_output.dtype, seq_length)
        encoder_extended_attention_mask = None
        head_mask = None




            






    
model = RegBert.from_pretrained('bert-base-uncased')
print(model.config)
for name in model.state_dict().keys():
    print(name)

# Block
# Feed Forward