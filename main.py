import argparse
import os
#from libs.solver import Solver
from libs.solver import Solver
from libs.solver_unet import Solver_unet
from datas.data_loader import get_loader, setup_seed
from torch.backends import cudnn
import random
from libs.utils import Logger
import sys
import torch
import numpy as np

def main(config):
    #cudnn.benchmark = True
    if config.model_type not in ['MSU_Net','MSAttU_Net','MSR2U_Net', 'MSR2AttU_Net', 'U_Net','AttU_Net','R2U_Net', 'R2AttU_Net']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
        print('Your input for model_type was %s'%config.model_type)
        return

    if config.ms_mode not in ['only_global', 'only_specific', '']:
        print('ERROR!! only_global/only_specific should use or not set')
        print('Your input for model_type was %s'%config.model_type)
        return


    print(config.lamda)
    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path,config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path, config.check_name)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    sys.stdout = Logger(config.result_path + "/log.txt")

    print(config)
        
    train_loader = get_loader(image_path=config.train_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            args = config,
                            num_workers=config.num_workers,
                            mode='train',
                            augmentation_prob=config.augmentation_prob)
    valid_loader = get_loader(image_path=config.valid_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            args = config,
                            num_workers=config.num_workers,
                            mode='valid',
                            augmentation_prob=0.)
    test_loader = get_loader(image_path=config.test_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            args = config,
                            num_workers=config.num_workers,
                            mode='test',
                            augmentation_prob=0.)

    if 'MS' in config.model_type:
        solver = Solver(config, train_loader, valid_loader, test_loader)
    else:
        solver = Solver_unet(config, train_loader, valid_loader, test_loader)

    
    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--t', type=int, default=2, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_epochs_decay', type=int, default=[80, 150])
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--ms_mode', type=str, default='', help='')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--train_path', type=str, default='/sda/DataSet/Beiyi/datasets/dataset_0113/train')
    parser.add_argument('--valid_path', type=str, default='/sda/DataSet/Beiyi/datasets/dataset_0113/test')
    parser.add_argument('--test_path', type=str, default='/sda/DataSet/Beiyi/datasets/dataset_0113/test')
    parser.add_argument('--result_path', type=str, default='./result/')
    parser.add_argument('--check_name', type=str, default='debug')

    parser.add_argument('--check_path', type=str, default='')

    parser.add_argument('--cuda_idx', type=int, default=0)

    parser.add_argument('--type', type=str, default='multitask') #coseg, choroid, blood
    parser.add_argument('--is_demo', action="store_true")
    parser.add_argument('--is_continuely', action="store_true")
    parser.add_argument('--resume_path', type=str, default='')

    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--alpha', type=float, default=1) #choroid
    parser.add_argument('--beta', type=float, default=0) #blood

    parser.add_argument('--seed', type=float, default=80) #blood
    parser.add_argument('--lamda', type=float, default=0.8) #blood

    parser.add_argument('--eval_frequency', type=float, default=20) #blood

    config = parser.parse_args()

    torch.manual_seed(config.seed)
    os.environ['PYTHONHASHSEED'] = str(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    
    main(config)
