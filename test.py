import torch
from torch.autograd import Variable
from argparse import ArgumentParser
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
from models.Unet import *
from utils.utils import *
from dataloader import *

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config_test.yaml', help="training configuration")


def find_fpr(fpr, tpr, tpr_target):
    min_dist = 10000
    min_idx = 0
    for i in range(len(tpr)):
        if tpr[i] > 0.95 and np.abs(tpr[i] - tpr_target) < min_dist:
            min_dist = np.abs(tpr[i] - tpr_target)
            min_idx = i

    print("fpr = {:.4f} for tpr = {:.4f}".format(fpr[min_idx], tpr[min_idx]))
    return fpr[min_idx], tpr[min_idx]


def test(model, normal_class, perm_list, perm_cost, test_dataloader):
    label_score_max = []
    label_score_min = []
    label_score_avg = []

    model.eval()

    for ind, data in enumerate(test_dataloader):
        print('{}/10000'.format(ind * test_dataloader.batch_size))
        inputs, labels = data
        target = inputs
        target = Variable(target).cuda()
        partitioned_img, base = split_tensor(inputs, tile_size=inputs.size(2) // 2, offset=inputs.size(2) // 2)

        min_score = torch.zeros(inputs.size(0)).cuda() + 100000
        avg_score = torch.zeros(inputs.size(0)).cuda()
        max_score = torch.zeros(inputs.size(0)).cuda()
        idx = 0
        num_perm = len(perm_list)
        for perm in perm_list:
            extended_perm = torch.tensor(perm) * inputs.size(1)
            if inputs.size(1) == 3:
                offset = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]])
                final_perm = offset + extended_perm[:, None]
                final_perm = final_perm.view(-1)
            else:
                final_perm = extended_perm

            permuted_img = partitioned_img[:, final_perm, :, :]
            permuted_img = rebuild_tensor(permuted_img, base, tile_size=inputs.size(2) // 2, offset=inputs.size(2) // 2)
            img = Variable(permuted_img).cuda()
            outputs = model(img)
            outputs = outputs.view(outputs.shape[0], -1)
            target = target.view(target.shape[0], -1)

            scores = torch.mean((target - outputs) ** 2, dim=1)
            scores /= torch.tensor(perm_cost[idx])

            min_score = torch.min(scores, min_score)
            max_score = torch.max(scores, max_score)
            avg_score += scores
            idx += 1

        avg_score = avg_score / num_perm

        label_score_avg += list(zip(labels.cpu().data.numpy().tolist(), avg_score.cpu().data.numpy().tolist()))
        label_score_max += list(zip(labels.cpu().data.numpy().tolist(), max_score.cpu().data.numpy().tolist()))
        label_score_min += list(zip(labels.cpu().data.numpy().tolist(), min_score.cpu().data.numpy().tolist()))

    label_scores = {'max': label_score_max, 'min': label_score_min, 'avg': label_score_avg}
    AUC = dict()

    for key, label_score in label_scores.items():
        labels, scores = zip(*label_score)
        labels = np.array(labels)
        indx1 = labels == normal_class
        indx2 = labels != normal_class
        labels[indx1] = 1
        labels[indx2] = 0
        scores = np.array(scores)
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=0)

        roc_auc = auc(fpr, tpr)
        roc_auc = round(roc_auc, 4)
        AUC[key] = roc_auc

    return AUC


def get_avg_val_error_per_permutation(model, permutation_list, val_dataloader):
    permutation_cost = []

    for ind, perm in enumerate(permutation_list):
        avg_score = 0
        for data in val_dataloader:
            inputs = data[0]
            orig_img = inputs
            target = orig_img
            target = Variable(target).cuda()
            partitioned_img, base = split_tensor(inputs, tile_size=inputs.size(2) // 2, offset=inputs.size(2) // 2)

            extended_perm = torch.tensor(perm) * inputs.size(1)
            if inputs.size(1) == 3:
                offset = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]])
                final_perm = offset + extended_perm[:, None]
                final_perm = final_perm.view(-1)
            else:
                final_perm = extended_perm
            permuted_img = partitioned_img[:, final_perm, :, :]

            permuted_img = rebuild_tensor(permuted_img, base, tile_size=inputs.size(2) // 2, offset=inputs.size(2) // 2)
            img = Variable(permuted_img).cuda()
            outputs = model(img)
            outputs = outputs.view(outputs.shape[0], -1)
            target = target.view(target.shape[0], -1)
            scores = torch.mean((target - outputs) ** 2)

            avg_score += scores.item()
        permutation_cost.append(avg_score)
    return permutation_cost


def main():
    args = parser.parse_args()
    config = get_config(args.config)

    n_channel = config['n_channel']
    normal_class = config["normal_class"]
    checkpoint_path = "outputs/{}/{}/checkpoints/".format(config['dataset_name'], normal_class)

    _, val_dataloader, test_dataloader = load_data(config)

    model = UNet(n_channel, n_channel).cuda()
    model.load_state_dict(torch.load(checkpoint_path + '{}.pth'.format(str(config['last_epoch']))))

    permutation_list = get_all_permutations()

    perm_cost = get_avg_val_error_per_permutation(model, permutation_list, val_dataloader)
    auc_dict = test(model, normal_class, permutation_list, perm_cost, test_dataloader)
    print(auc_dict)


if __name__ == '__main__':
    main()
