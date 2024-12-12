import torch
import numpy as np 
import torch.nn as nn
# from transformers.models.bert.modeling_bert import BertModel, BertLayer, BertSelfAttention, BaseModelOutputWithPoolingAndCrossAttentions, QuestionAnsweringModelOutput, BertForQuestionAnswering
from BERT import BertModel
from layers_ours import Linear ##
from transformers.models.bert.modeling_bert import QuestionAnsweringModelOutput, BertForQuestionAnswering, BaseModelOutputWithPoolingAndCrossAttentions
import math
from typing import List, Optional, Tuple, Union
import warnings


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


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor



class RegBert(BertModel):
    def __init__(self, config, num_registers=50, dev='cuda'):
        
        super().__init__(config)
        self.num_registers = num_registers
        self.config = config
        if self.num_registers > 0: 
            self.reg_tokens = nn.Parameter(torch.zeros(1, num_registers, self.config.hidden_size))
            self.reg_pos = nn.Parameter(torch.zeros(1, num_registers, self.config.hidden_size))
        self.dev = dev
        trunc_normal_(self.reg_tokens, std=.02)
        trunc_normal_(self.reg_pos, std=.02)
        # self.init_reg_weights()
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, num_registers=50, dev='cuda', **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        # Add the `num_registers` attribute and initialize register tokens
        model.num_registers = num_registers
        if num_registers > 0:
            model.reg_tokens = nn.Parameter(torch.zeros(1, num_registers, model.config.hidden_size))
            model.reg_pos = nn.Parameter(torch.zeros(1, num_registers, model.config.hidden_size))
        model.dev = dev
        trunc_normal_(model.reg_tokens, std=.02)
        trunc_normal_(model.reg_pos, std=.02)
        return model


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        # past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None):

        ones = torch.ones((attention_mask.shape[0], self.num_registers)).to(self.dev)
        attention_mask = torch.cat((ones, attention_mask), dim =1)
        batch_size, seq_length = input_ids.size()
        input_shape = input_ids.size()
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.dev)
        
        # Here are the positional embeddings + word embeddings + token type embeddings
        input_embeds = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        # print("Forwarding")
        if self.num_registers > 0: 
            register = self.reg_tokens.expand(batch_size, -1, -1)
            register = torch.add(register, self.reg_pos)
            embedding_output = torch.cat((register, input_embeds), dim=1)
        else:
            embedding_output = input_embeds

        #prepare 4D attention mask if needed
        def prepare_mask(mask, dtype, target_length):
            batch, key_length = mask.shape 
            target_length = target_length if target_length is not None else key_length
            expanded_mask = mask[:, None, None, :].expand(batch, 1, target_length, key_length).to(self.dev)
            inverted_mask = 1.0 - expanded_mask
            return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)
        extended_attention_mask = prepare_mask(attention_mask, embedding_output.dtype, seq_length+self.num_registers)
        # print(extended_attention_mask.shape)
        encoder_extended_attention_mask = None
        head_mask = None


        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            # past_key_values=past_key_values,
            # use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            # past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            # cross_attentions=encoder_outputs.cross_attentions,
        )

class RegBertForQA(BertForQuestionAnswering):

    def __init__(self, config, num_registers=50):
        super().__init__(config)
        self.num_labels = config.num_labels

        # self.bert = RegBert(config)
        print('from regbertfor QA, num_reg=', num_registers)
        self.bert = RegBert.from_pretrained('bert-base-uncased', num_registers=num_registers)
        # self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.qa_outputs = Linear(config.hidden_size, config.num_labels)
        self._logits = []
        # Initialize weights and apply final processing
        self.post_init() ####### Do we need this??

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor]]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # print('outputs[0]',outputs[0].shape)

        sequence_output = outputs[0]
        self._logits.append(sequence_output)

        logits = self.qa_outputs(sequence_output)
        # print('logits: ',logits.shape)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        start_logits = start_logits[:, self.bert.num_registers:]
        end_logits = end_logits[:, self.bert.num_registers:]

        # print('start_logits: ',start_logits.shape)
        # print('end_logits: ', end_logits.shape)
        # print('start_positions: ',start_positions.shape)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def relprop(self, cam=None, **kwargs):
        # cam = self.qa_outputs.relprop(cam, **kwargs)
        # cam = self.dropout.relprop(cam, **kwargs)
        cam = self.bert.relprop(cam, **kwargs)
        # print("conservation: ", cam.sum())
        return cam


# model = RegBert.from_pretrained('bert-base-uncased')
# model.init_reg_weights()
# print(model.config)
# for name, value in model.state_dict().items():
#     print(name, value.shape)


# Block
# Feed Forward