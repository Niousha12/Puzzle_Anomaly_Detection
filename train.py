from torch import nn
from random import randrange
from models.Discriminator import *
from test import *
from dataloader import *
from pathlib import Path

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config_train.yaml', help="training configuration")


def main():
    args = parser.parse_args()
    config = get_config(args.config)

    n_channel = config['n_channel']
    normal_class = config["normal_class"]
    dataset_name = config['dataset_name']

    checkpoint_path = "outputs/{}/{}/checkpoints/".format(dataset_name, normal_class)
    train_output_path = "outputs/{}/{}/train_outputs/".format(dataset_name, normal_class)

    # create directory
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    Path(train_output_path).mkdir(parents=True, exist_ok=True)

    epsilon = float(config['eps'])
    alpha = float(config['alpha'])

    get_random_permutation = get_unforced_random_permutation

    # if dataset_name == 'MVTec':
    #     get_random_permutation = get_forced_random_permutation
    # else:
    #     get_random_permutation = get_unforced_random_permutation

    train_dataloader, _, _ = load_data(config)

    unet = UNet(n_channel, n_channel, config['base_channel']).cuda()
    discriminator = NetD(config['image_size'], n_channel, config['n_extra_layers']).cuda()
    discriminator.apply(weights_init)

    criterion = nn.MSELoss()
    optimizer_u = torch.optim.Adam(
        unet.parameters(), lr=config['lr_u'], weight_decay=float(config['weight_decay']))

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=config['lr_d'], betas=(config['beta1'], config['beta2']))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_u, factor=config['factor'], patience=config['patience'], mode='min', verbose=True)

    ae_loss_list = []

    l_adv = l2_loss
    l_bce = nn.BCELoss()

    num_epochs = config['num_epochs']
    epoch_loss_dict = dict()
    unet.train()
    discriminator.train()

    for epoch in range(num_epochs + 1):
        epoch_ae_loss = 0
        epoch_total_loss = 0

        for data in train_dataloader:
            rand_number = randrange(4)
            img = data[0]
            orig_img = img

            partitioned_img, base = split_tensor(img, tile_size=img.size(2) // 2, offset=img.size(2) // 2)
            perm = get_random_permutation()

            extended_perm = perm * img.size(1)
            if img.size(1) == 3:
                offset = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]])
                final_perm = offset + extended_perm[:, None]
                final_perm = final_perm.view(-1)
            else:
                final_perm = extended_perm

            permuted_img = partitioned_img[:, final_perm, :, :]

            if img.size(1) == 3:
                avg = permuted_img[:, rand_number * 3, :, :] + permuted_img[:, rand_number * 3 + 1, :, :] + \
                      permuted_img[:, rand_number * 3 + 2, :, :]

                avg /= 3
                permuted_img[:, rand_number * 3, :, :] = avg
                permuted_img[:, rand_number * 3 + 1, :, :] = avg
                permuted_img[:, rand_number * 3 + 2, :, :] = avg
            else:
                permuted_img[:, rand_number, :, :] *= 0

            target = orig_img
            permuted_img = rebuild_tensor(permuted_img, base, tile_size=img.size(2) // 2, offset=img.size(2) // 2)

            permuted_img = fgsm_attack(permuted_img, unet, eps=epsilon, alpha=alpha)

            img = Variable(permuted_img).cuda()
            target = Variable(target).cuda()

            # ===================forward=====================

            # Forward Unet
            output = unet(img)

            # Forward Discriminator
            pred_real, feat_real = discriminator(target)
            pred_fake, feat_fake = discriminator(output.detach())

            # ===================backward====================

            # Backward Unet
            optimizer_u.zero_grad()
            err_g_adv = l_adv(discriminator(target)[1], discriminator(output)[1])
            AE_loss = criterion(output, target)
            loss = config['adv_coeff'] * err_g_adv + AE_loss

            epoch_total_loss += loss.item()
            epoch_ae_loss += AE_loss.item()
            loss.backward()
            optimizer_u.step()

            # Backward Discriminator
            real_label = torch.ones(size=(img.shape[0],), dtype=torch.float32).cuda()
            fake_label = torch.zeros(size=(img.shape[0],), dtype=torch.float32).cuda()

            optimizer_d.zero_grad()
            err_d_real = l_bce(pred_real, real_label)
            err_d_fake = l_bce(pred_fake, fake_label)

            err_d = (err_d_real + err_d_fake) * 0.5
            err_d.backward()
            optimizer_d.step()

        # ===================log========================
        ae_loss_list.append(epoch_ae_loss)
        scheduler.step(epoch_ae_loss)

        print('epoch [{}/{}], epoch_total_loss:{:.4f}, epoch_ae_loss:{:.4f}, err_adv:{:.4f}'
              .format(epoch + 1, num_epochs, epoch_total_loss, epoch_ae_loss, err_g_adv))

        with open(checkpoint_path + 'log_{}.txt'.format(normal_class), "a") as log_file:
            log_file.write('\n epoch [{}/{}], loss:{:.4f}, epoch_ae_loss:{:.4f}, err_adv:{:.4f}'
                           .format(epoch + 1, num_epochs, epoch_total_loss, epoch_ae_loss, err_g_adv))

        if epoch % 500 == 0:
            show_process_for_trainortest(img, output, orig_img, train_output_path + str(epoch))
            epoch_loss_dict[epoch] = epoch_total_loss

            torch.save(unet.state_dict(), checkpoint_path + '{}.pth'.format(str(epoch)))
            torch.save(discriminator.state_dict(), checkpoint_path + 'netd_{}.pth'.format(str(epoch)))

            torch.save(optimizer_u.state_dict(), checkpoint_path + 'opt_{}.pth'.format(str(epoch)))
            torch.save(optimizer_d.state_dict(), checkpoint_path + 'optd_{}.pth'.format(str(epoch)))

            torch.save(scheduler.state_dict(), checkpoint_path + 'scheduler_{}.pth'.format(str(epoch)))


if __name__ == '__main__':
    main()
