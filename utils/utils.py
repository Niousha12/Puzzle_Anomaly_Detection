import torch
from torch.distributions import uniform
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math
import yaml


# get configs
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def get_mvtec_class_type(class_name):
    mvtec_class_type = {'bottle': 'object', 'cable': 'object', 'capsule': 'object', 'carpet': 'texture',
                        'grid': 'texture', 'hazelnut': 'object', 'leather': 'texture', 'metal_nut': 'object',
                        'pill': 'object', 'screw': 'object', 'tile': 'texture', 'toothbrush': 'object',
                        'transistor': 'object', 'wood': 'texture', 'zipper': 'object'
                        }
    return mvtec_class_type[class_name]


def fgsm_attack(inputs, model, eps=0.1, alpha=2):
    distribution = uniform.Uniform(torch.Tensor([-eps]), torch.Tensor([eps]))
    delta = distribution.sample(inputs.shape)
    delta = torch.squeeze(delta).reshape(-1, inputs.size(1), inputs.size(2), inputs.size(3))
    delta = delta.cuda()
    inputs = inputs.cuda()
    ori_inputs = inputs
    inputs += delta
    inputs = torch.clamp(inputs, min=0, max=1)
    inputs.requires_grad = True
    outputs = model(inputs)
    model.zero_grad()
    loss_rec = torch.mean(torch.sum((outputs - ori_inputs) ** 2, dim=1))
    loss_rec.backward()
    delta = delta + alpha * inputs.grad.sign()
    delta = torch.clamp(delta, min=-eps, max=eps)
    adv_inputs = inputs + delta
    adv_inputs = torch.clamp(adv_inputs, min=0, max=1).detach_()
    return adv_inputs


def show_process_for_trainortest(input_img, recons_img, puzzled_img=None, path="./"):
    if input_img.shape[0] > 15:
        n = 15
    else:
        n = input_img.shape[0]

    channel = input_img.shape[1]

    # print("Inputs:")
    show(np.transpose(input_img[0:n].cpu().detach().numpy(), (0, 2, 3, 1)), channel=channel, path=path + "_input.png")
    # print("Puzzle Input:")
    show(np.transpose(puzzled_img[0:n].cpu().detach().numpy(), (0, 2, 3, 1)), channel=channel,
         path=path + "_puzzle_input.png")
    # print("Reconstructions:")
    show(np.transpose(recons_img[0:n].cpu().detach().numpy(), (0, 2, 3, 1)), channel=channel,
         path=path + "_reconstruction.png")


def show(image_batch, rows=1, channel=3, path="./test.png"):
    # Set Plot dimensions
    cols = np.ceil(image_batch.shape[0] / rows)
    plt.rcParams['figure.figsize'] = (0.0 + cols, 0.0 + rows)  # set default size of plots

    for i in range(image_batch.shape[0]):
        plt.subplot(rows, cols, i + 1)
        if channel != 1:
            plt.imshow(image_batch[i])
        else:
            plt.imshow(image_batch[i].reshape(image_batch.shape[-2], image_batch.shape[-2]), cmap='gray')
        plt.axis('off')
    plt.savefig(path)


def split_tensor(tensor, tile_size=14, offset=14):
    c, h, w = tensor.size(1), tensor.size(2), tensor.size(3)
    col_count = int(math.ceil(h / offset))
    row_count = int(math.ceil(w / offset))
    tiles_count = col_count * row_count
    tiles = torch.zeros(tensor.size(0), tiles_count * c, tile_size, tile_size)
    for y in range(col_count):
        for x in range(row_count):
            ind = x + (y * row_count)
            tiles[:, ind * c:(ind + 1) * c, :, :] = tensor[:, :, offset * y:min(offset * y + tile_size, h),
                                                    offset * x:min(offset * x + tile_size, w)]
    if tensor.is_cuda:
        base_tensor = torch.zeros(tensor.size(), device=tensor.get_device())
    else:
        base_tensor = torch.zeros(tensor.size())
    return tiles, base_tensor


def rebuild_tensor(tensor_list, base_tensor, tile_size=14, offset=14):
    num_tiles = 0
    c, h, w = base_tensor.size(1), base_tensor.size(2), base_tensor.size(3)
    for y in range(int(math.ceil(h / offset))):
        for x in range(int(math.ceil(w / offset))):
            base_tensor[:, :, offset * y:min(offset * y + tile_size, h),
            offset * x:min(offset * x + tile_size, w)] = tensor_list[:, (num_tiles * c):((num_tiles + 1) * c), :, :]
            num_tiles += 1
    return base_tensor


def get_unforced_random_permutation():
    while True:
        perm = torch.randperm(4)
        if perm.tolist() != [0, 1, 2, 3]:
            return perm


def get_forced_random_permutation():
    while True:
        perm = torch.randperm(4)
        count = 0
        for i in range(len(perm)):
            if perm[i] != i: count += 1
        if count == 2:
            return perm


def get_all_permutations():
    # Get all permutations of 4 partitions
    permutation_list = list(itertools.permutations([0, 1, 2, 3]))
    permutation_list = [list(perm) for perm in permutation_list]
    permutation_list.remove([0, 1, 2, 3])
    return permutation_list
