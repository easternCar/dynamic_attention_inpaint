import os
import random
import time
import shutil
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from trainer import Trainer

from data.dataset import Dataset
import torchvision.transforms as transforms
from utils.tools import get_config
from utils.mask_tools import mask_image_predefined, load_predefined_mask
from utils.logger import get_logger

import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="training configuration")
parser.add_argument('--seed', type=int, help='manual seed')



def main():
    args = parser.parse_args()
    config = get_config(args.config)


    # -------------======= CENTER MASK & FREE MASK SETTING ==========-----------------
    config['checkpoints'] = 'check_' + config['dataset_name'].lower() + '_' + config['mask_type'].lower()
    # ---------------==================================================----------------

    # CUDA configuration
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    orig_gpus = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True


    checkpoint_path = config['checkpoints']
    #checkpoint_path = os.path.join(config['checkpoints'],
    #                               config['dataset_name'],
    #                               config['mask_type'] + '_' + mask_shape_type + '_' + config['expname'])
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    shutil.copy(args.config, os.path.join(checkpoint_path, os.path.basename(args.config)))
    logger = get_logger(checkpoint_path)    # get logger and configure it at the first call

    logger.info(f"Arguments: {args}")
    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    logger.info(f"Random seed: {args.seed}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)

    # Log the configuration
    logger.info(f"Configuration: {config}")

    try:  # for unexpected error logging
        # Load the dataset
        logger.info(f"Training on dataset: {config['dataset_name']}")
        train_transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])


        train_dataset = Dataset(data_path=config['data_path'], transform=train_transform, \
            image_shape=config['image_shape'], with_subfolder=config['data_with_subfolder'])
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=config['batch_size'],
                                                    shuffle=True,
                                                    num_workers=config['num_workers'], drop_last=False)

        '''
        if config['dataset_name'] == 'CELEBA':
            print("CASIA START")
            train_dataset = CASIAWebFace(root=config['train_data_path'], transform=train_transform, img_size=config['image_shape'][1],
                                         file_list=config['train_data_list_file'])
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=config['batch_size'],
                                                       shuffle=True,
                                                       num_workers=config['num_workers'], drop_last=False)
            print("CASIA WEB FINISHED")
        elif config['dataset_name'] == "PLACE2":
            train_dataset = MS_Celeb_1M(config['train_data_path'], config['train_data_list_file'], transform=train_transform, img_size=config['image_shape'][1])
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'],
                                                      shuffle=True, num_workers=config['num_workers'], drop_last=False)
        '''


        # Define the trainer
        trainer = Trainer(config)

        # CUDA AVAILABLE
        if cuda:
            trainer = nn.parallel.DataParallel(trainer, device_ids=device_ids)
            trainer_module = trainer.module
        else:
            trainer_module = trainer


        # Get the resume iteration to restart training
        if config['resume']:
            start_iteration = trainer_module.resume(config['resume'], iteration=int(config['resume_iter']))
        else:
            start_iteration = 1


        iterable_train_loader = iter(train_loader)
        time_count = time.time()

        for iteration in range(start_iteration, config['niter'] + 1):

            # -=--------------------------------- TRAIN -------------------
            # get gt from TRAIN LOADER
            try:
                train_imgs = iterable_train_loader.next()
            except StopIteration:
                iterable_train_loader = iter(train_loader)
                train_imgs = iterable_train_loader.next()

            # Prepare the inputs
            occ_mask = load_predefined_mask(config, batch_size=train_imgs.size(0))
            x, mask = mask_image_predefined(config, train_imgs, occ_mask)
            
            if cuda:
                x = x.cuda()
                mask = mask.cuda()
                train_imgs = train_imgs.cuda()


            ###### Forward pass ######
            losses, _, x2_inpainted_result = trainer(x, mask, train_imgs)


            # Scalars from different devices are gathered into vectors
            for k in losses.keys():
                if not losses[k].dim() == 0:
                    losses[k] = torch.mean(losses[k])

            ###### Backward pass ######
            # Update D -----------------------------------
            trainer_module.optimizer_d.zero_grad()
            losses['d'] = losses['wgan_d']
            losses['d'] = losses['d'] + losses['wgan_gp'] * config['wgan_gp_lambda']
            losses['d'].backward()
            trainer_module.optimizer_d.step()


            # Update G -----------------------------------
            trainer_module.optimizer_g.zero_grad()

            losses['g'] = losses['l1'] * config['l1_loss_alpha'] \
                          + losses['wgan_g'] * config['gan_loss_alpha'] \
                          + losses['att'] * config['attention_alpha'] \
                          + losses['percep'] * config['percep_alpha']

            losses['g'].backward()
            trainer_module.optimizer_g.step()

            # ------------ Log and visualization
            log_losses = ['l1', 'wgan_g', 'wgan_d', 'wgan_gp', 'g', 'd', 'att', 'percep']

            if iteration % config['print_iter'] == 0:
                trainer_module.eval()   # --- eval

                time_count = time.time() - time_count
                speed = config['print_iter'] / time_count
                speed_msg = 'speed: %.2f batches/s ' % speed
                time_count = time.time()

                MASK_TYPE = config['mask_type'].upper()
                
                message = str(orig_gpus) + '/' + config['dataset_name'] + '/' + MASK_TYPE + \
                    'Iter: [%d/%d] ' % (iteration, config['niter'])
                for k in log_losses:
                    v = losses[k]
                    message += '%s: %.6f ' % (k, v)
                message += speed_msg
                logger.info(message)

                trainer_module.train()     # --- eval off


            if iteration % (config['viz_iter']) == 0:
                trainer_module.eval()   # --- eval

                viz_max_out = config['viz_max_out']
                if x.size(0) > viz_max_out:
                    viz_images = torch.stack([x[:viz_max_out], x2_inpainted_result[:viz_max_out]], dim=1)
                else:
                    #viz_images = torch.stack([x, inpainted_result, offset_flow], dim=1)
                    viz_images = torch.stack([x, x2_inpainted_result], dim=1)
                
                viz_images = viz_images.view(-1, *list(x.size())[1:])
                
                vutils.save_image(viz_images,
                                  '%s/niter_%03d.png' % (checkpoint_path, iteration),
                                  nrow=3 * 4,
                                  normalize=True)

                trainer_module.train()     # --- eval off


            # Save the model
            if iteration % config['snapshot_save_iter'] == 0:
                trainer_module.save_model(checkpoint_path, iteration)



    except Exception as e:  # for unexpected error logging
        logger.error(f"{e}")
        raise e


if __name__ == '__main__':
    main()
