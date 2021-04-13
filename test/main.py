import torch
import torch.nn as nn

channel_count = 6
eps = 1e-3
batch_size = 20
input_size = 2

input_tensor = torch.randn(batch_size, channel_count, input_size)


def custom_group_norm(input_tensor, groups, eps):
    normed_tensor = torch.zeros(input_tensor.shape)
    shift = input_tensor.shape[1] // groups
    for batch in range(input_tensor.shape[0]):
        for channels in range(0,input_tensor.shape[1],shift):
            temp = input_tensor[batch,channels:channels + shift]
            mean = temp.mean()
            var = temp.var(unbiased=False)
            normed_tensor[batch,channels:channels + shift] = (temp - mean) / torch.sqrt(var + eps)
    return normed_tensor


all_correct = True
for groups in [2]:
    group_norm = nn.GroupNorm(groups, channel_count, eps=eps, affine=False)
    norm_output = group_norm(input_tensor)
    custom_output = custom_group_norm(input_tensor, groups, eps)
    all_correct &= torch.allclose(norm_output, custom_output, 1e-3)
    all_correct &= norm_output.shape == custom_output.shape
print(all_correct)