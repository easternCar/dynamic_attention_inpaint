import os
import random
import time
import shutil
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torch.nn import Parameter

from model.network import Generator

from data.dataset import Dataset
import torchvision.transforms as transforms
from utils.tools import get_config, save_each_image_everything
from utils.mask_tools import mask_image_predefined, load_predefined_mask
from utils.tools import get_model_list


# load config
parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="training configuration")
parser.add_argument('--seed', type=int, help='manual seed')


def load_my_state_dict(our_model, state_dict):
    own_state = our_model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        own_state[name].copy_(param)

def main():
    args = parser.parse_args()
    config = get_config(args.config)

    # CUDA configuration
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    orig_gpus = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True

    # checkpoint path existing check
    checkpoint_path = config['test_checkpoints'] 
    if not os.path.exists(checkpoint_path):
        print("Not existing checkpoint path for inference")
        return

    # saveing path generate
    output_path = config['test_save_path']
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random seed: {}".format(args.seed))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)

    test_transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])


    test_dataset = Dataset(data_path=config['test_data_path'], transform=test_transform, \
        image_shape=config['image_shape'], with_subfolder=config['data_with_subfolder'], return_name=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=config['test_batch_size'],
                                                shuffle=True,
                                                num_workers=config['num_workers'], drop_last=False)

    # load Generator
    netG = Generator(config['netG'])
    # Resume weight
    last_model_name = get_model_list(checkpoint_path, "gen", iteration=int(config['test_resume_iter']))
    load_my_state_dict(netG, torch.load(last_model_name))

    # CUDA AVAILABLE
    if cuda:
        netG = nn.parallel.DataParallel(netG, device_ids=device_ids)
        #trainer = nn.parallel.DataParallel(trainer, device_ids=device_ids)
        #trainer_module = trainer.module
    #else:
        #trainer_module = trainer


    iterable_test_loader = iter(test_loader)
    time_count = time.time()

    # --------------------- <INFERENCE> --------------------------
    for test_batch in iterable_test_loader:
        with torch.no_grad():
            #test_names, test_imgs = iterable_test_loader.next()
            test_names, test_imgs = test_batch

            # Prepare the inputs
            occ_mask = load_predefined_mask(config, batch_size=test_imgs.size(0))
            x, mask = mask_image_predefined(config, test_imgs, occ_mask)
            
            if cuda:
                x = x.cuda()
                mask = mask.cuda()
                test_imgs = test_imgs.cuda()

            ###### Forward pass ######
            x2_out, x1_out, daw_maps = netG(x, mask)
            result = x2_out * mask + x * (1. - mask)

            # SAVE #
            save_each_image_everything(result, output_path, test_names, normalize=False)

if __name__ == '__main__':
    main()
