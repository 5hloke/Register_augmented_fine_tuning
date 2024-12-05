import argparse
import numpy as np
import torch
import glob

# compute rollout between attention layers
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention

class Generator:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def generate_LRP(self, input_ids, attention_mask,
                     start_index=None, end_index=None, start_layer=11):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        start_logits, end_logits = outputs[0], outputs[1]
        outputs = self.model._logits[0]
        kwargs = {"alpha": 1}
        print(f"outputs: {outputs.shape}")
        
        index = np.argmax(outputs.cpu().detach().numpy(), axis = -1) 
        
        print(f"Start logits shape: {start_logits.shape}")
        print(f"End logits shape: {end_logits.shape}")
        # start position
        one_hot = np.zeros((1, outputs.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * outputs)
        # one_hot_start = torch.zeros_like(start_logits)
        # one_hot_start[0, start_index] = 1
        # one_hot_start = one_hot_start.requires_grad_(True)
        # one_hot_output_start = torch.sum(one_hot_start * start_logits)

        self.model.zero_grad()
        # one_hot_output_start.backward(retain_graph=True)
        one_hot.backward(retain_graph=True)
        
        # print(f"One hot start shape: {one_hot_start.shape}")
        # print(f" Start logits are: {start_logits}")
        # print(f" End logits are: {end_logits}")
        # print(f"One hot end shape: {one_hot_end.shape}")
        self.model.relprop(torch.tensor(one_hot).to(input_ids.device), **kwargs)

        cams_start = []

        blocks = self.model.bert.encoder.layer
        for blk in blocks:
            grad = blk.attention.self.get_attn_gradients()
            cam = blk.attention.self.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cams_start.append(cam.unsqueeze(0))
        rollout_start = compute_rollout_attention(cams_start, start_layer=start_layer)
        rollout_start[:, 0, 0] = rollout_start[:, 0].min()

#         # end position
#         one_hot_end = torch.zeros_like(end_logits)
#         one_hot_end[0, end_index] = 1
#         one_hot_end = one_hot_end.requires_grad_(True)
#         one_hot_output_end = torch.sum(one_hot_end * end_logits)

#         self.model.zero_grad()
#         one_hot_output_end.backward(retain_graph=True)
#         self.model.relprop(torch.tensor(one_hot_end).to(input_ids.device), **kwargs)

#         cams_end = []
#         for blk in blocks:
#             grad = blk.attention.self.get_attn_gradients()
#             cam = blk.attention.self.get_attn_cam()
#             cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
#             grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
#             cam = grad * cam
#             cam = cam.clamp(min=0).mean(dim=0)
#             cams_end.append(cam.unsqueeze(0))
#         rollout_end = compute_rollout_attention(cams_end, start_layer=start_layer)
#         rollout_end[:, 0, 0] = rollout_end[:, 0].min()

        # Average the rollouts
        # rollout = (rollout_start + rollout_end) / 2
        return rollout_start[:, 0]

#     def generate_LRP_last_layer(self, input_ids, attention_mask,
#                                 start_index=None, end_index=None):
#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
#         start_logits, end_logits = outputs[0], outputs[1]
#         kwargs = {"alpha": 1}

#         if start_index is None:
#             start_index = torch.argmax(start_logits, dim=-1).item()

#         if end_index is None:
#             end_index = torch.argmax(end_logits, dim=-1).item()

#         # Process start position
#         one_hot_start = torch.zeros_like(start_logits)
#         one_hot_start[0, start_index] = 1
#         one_hot_start = one_hot_start.requires_grad_(True)
#         one_hot_output_start = torch.sum(one_hot_start * start_logits)

#         self.model.zero_grad()
#         one_hot_output_start.backward(retain_graph=True)
#         self.model.relprop(torch.tensor(one_hot_start).to(input_ids.device), **kwargs)

#         cam_start = self.model.bert.encoder.layer[-1].attention.self.get_attn_cam()[0]
#         cam_start = cam_start.clamp(min=0).mean(dim=0).unsqueeze(0)
#         cam_start[:, 0, 0] = 0

#         # Process end position
#         one_hot_end = torch.zeros_like(end_logits)
#         one_hot_end[0, end_index] = 1
#         one_hot_end = one_hot_end.requires_grad_(True)
#         one_hot_output_end = torch.sum(one_hot_end * end_logits)

#         self.model.zero_grad()
#         one_hot_output_end.backward(retain_graph=True)
#         self.model.relprop(torch.tensor(one_hot_end).to(input_ids.device), **kwargs)

#         cam_end = self.model.bert.encoder.layer[-1].attention.self.get_attn_cam()[0]
#         cam_end = cam_end.clamp(min=0).mean(dim=0).unsqueeze(0)
#         cam_end[:, 0, 0] = 0

#         # Average cams
#         cam = (cam_start + cam_end) / 2
#         return cam[:, 0]

#     def generate_full_lrp(self, input_ids, attention_mask,
#                           start_index=None, end_index=None):
#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
#         start_logits, end_logits = outputs[0], outputs[1]
#         kwargs = {"alpha": 1}

#         if start_index is None:
#             start_index = torch.argmax(start_logits, dim=-1).item()

#         if end_index is None:
#             end_index = torch.argmax(end_logits, dim=-1).item()

#         # start position
#         one_hot_start = torch.zeros_like(start_logits)
#         one_hot_start[0, start_index] = 1
#         one_hot_start = one_hot_start.requires_grad_(True)
#         one_hot_output_start = torch.sum(one_hot_start * start_logits)

#         self.model.zero_grad()
#         one_hot_output_start.backward(retain_graph=True)
#         cam_start = self.model.relprop(torch.tensor(one_hot_start).to(input_ids.device), **kwargs)
#         cam_start = cam_start.sum(dim=2)
#         cam_start[:, 0] = 0

#         # end position
#         one_hot_end = torch.zeros_like(end_logits)
#         one_hot_end[0, end_index] = 1
#         one_hot_end = one_hot_end.requires_grad_(True)
#         one_hot_output_end = torch.sum(one_hot_end * end_logits)

#         self.model.zero_grad()
#         one_hot_output_end.backward(retain_graph=True)
#         cam_end = self.model.relprop(torch.tensor(one_hot_end).to(input_ids.device), **kwargs)
#         cam_end = cam_end.sum(dim=2)
#         cam_end[:, 0] = 0

#         # Average cams
#         cam = (cam_start + cam_end) / 2
#         return cam

#     def generate_attn_last_layer(self, input_ids, attention_mask,
#                                  start_index=None, end_index=None):
#         # Attention maps are independent of output in this case
#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
#         cam = self.model.bert.encoder.layer[-1].attention.self.get_attn()[0]
#         cam = cam.mean(dim=0).unsqueeze(0)
#         cam[:, 0, 0] = 0
#         return cam[:, 0]

#     def generate_rollout(self, input_ids, attention_mask, start_layer=0, start_index=None, end_index=None):
#         self.model.zero_grad()
#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
#         blocks = self.model.bert.encoder.layer
#         all_layer_attentions = []
#         for blk in blocks:
#             attn_heads = blk.attention.self.get_attn()
#             avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
#             all_layer_attentions.append(avg_heads)
#         rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
#         rollout[:, 0, 0] = 0
#         return rollout[:, 0]

#     def generate_attn_gradcam(self, input_ids, attention_mask, start_index=None, end_index=None):
#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
#         start_logits, end_logits = outputs[0], outputs[1]
#         kwargs = {"alpha": 1}

#         if start_index is None:
#             start_index = torch.argmax(start_logits, dim=-1).item()

#         if end_index is None:
#             end_index = torch.argmax(end_logits, dim=-1).item()

#         # Process start position
#         one_hot_start = torch.zeros_like(start_logits)
#         one_hot_start[0, start_index] = 1
#         one_hot_start = one_hot_start.requires_grad_(True)
#         one_hot_output_start = torch.sum(one_hot_start * start_logits)

#         self.model.zero_grad()
#         one_hot_output_start.backward(retain_graph=True)
#         self.model.relprop(torch.tensor(one_hot_start).to(input_ids.device), **kwargs)

#         cam_start = self.model.bert.encoder.layer[-1].attention.self.get_attn()
#         grad_start = self.model.bert.encoder.layer[-1].attention.self.get_attn_gradients()
#         cam_start = cam_start[0].reshape(-1, cam_start.shape[-1], cam_start.shape[-1])
#         grad_start = grad_start[0].reshape(-1, grad_start.shape[-1], grad_start.shape[-1])
#         grad_start = grad_start.mean(dim=[1, 2], keepdim=True)
#         cam_start = (cam_start * grad_start).mean(0).clamp(min=0).unsqueeze(0)
#         cam_start = (cam_start - cam_start.min()) / (cam_start.max() - cam_start.min())
#         cam_start[:, 0, 0] = 0

#         # Process end position
#         one_hot_end = torch.zeros_like(end_logits)
#         one_hot_end[0, end_index] = 1
#         one_hot_end = one_hot_end.requires_grad_(True)
#         one_hot_output_end = torch.sum(one_hot_end * end_logits)

#         self.model.zero_grad()
#         one_hot_output_end.backward(retain_graph=True)
#         self.model.relprop(torch.tensor(one_hot_end).to(input_ids.device), **kwargs)

#         cam_end = self.model.bert.encoder.layer[-1].attention.self.get_attn()
#         grad_end = self.model.bert.encoder.layer[-1].attention.self.get_attn_gradients()
#         cam_end = cam_end[0].reshape(-1, cam_end.shape[-1], cam_end.shape[-1])
#         grad_end = grad_end[0].reshape(-1, grad_end.shape[-1], grad_end.shape[-1])
#         grad_end = grad_end.mean(dim=[1, 2], keepdim=True)
#         cam_end = (cam_end * grad_end).mean(0).clamp(min=0).unsqueeze(0)
#         cam_end = (cam_end - cam_end.min()) / (cam_end.max() - cam_end.min())
#         cam_end[:, 0, 0] = 0

#         # Average cams
#         cam = (cam_start + cam_end) / 2
#         return cam[:, 0]
