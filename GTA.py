import os
import time
import torch
import argparse
import numpy as np
from inference import infer, save_mel
from utils.util import mode
from hparams import hparams as hps
from utils.logger import Tacotron2Logger
from utils.dataset import ljdataset, ljcollate
from model.model import Tacotron2, Tacotron2Loss
from torch.utils.data import DistributedSampler, DataLoader
np.random.seed(hps.seed)
torch.manual_seed(hps.seed)
torch.cuda.manual_seed(hps.seed)


def prepare_dataloaders(fdir, n_gpu):
    trainset = ljdataset(fdir)
    collate_fn = ljcollate(hps.n_frames_per_step)
    sampler = DistributedSampler(trainset) if n_gpu > 1 else None
    train_loader = DataLoader(trainset, num_workers = 4, shuffle = n_gpu == 1,
                              batch_size = 32, pin_memory = True,
                              drop_last = False, collate_fn = collate_fn, sampler = sampler)
    return train_loader


def load_checkpoint(ckpt_pth, model, optimizer, device, n_gpu):
    ckpt_dict = torch.load(ckpt_pth, map_location = device)
    (model.module if n_gpu > 1 else model).load_state_dict(ckpt_dict['model'])
    optimizer.load_state_dict(ckpt_dict['optimizer'])
    iteration = ckpt_dict['iteration']
    return model, optimizer, iteration


def save_checkpoint(model, optimizer, iteration, ckpt_pth, n_gpu):
    torch.save({'model': (model.module if n_gpu > 1 else model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': iteration}, ckpt_pth)


def GTA(args):
    # setup env
    rank = local_rank = 0
    n_gpu = 1
    if 'WORLD_SIZE' in os.environ:
        os.environ['OMP_NUM_THREADS'] = str(hps.n_workers)
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        n_gpu = int(os.environ['WORLD_SIZE'])
        torch.distributed.init_process_group(
            backend = 'nccl', rank = local_rank, world_size = n_gpu)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:{:d}'.format(local_rank))

    # build model
    model = Tacotron2()
    mode(model, True)
    if n_gpu > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids = [local_rank])
    
    # load checkpoint
    iteration = 1
    if args.ckpt_pth != '':
        model, optimizer, iteration = load_checkpoint(args.ckpt_pth, model, optimizer, device, n_gpu)
        iteration += 1
    
    # make dataset
    train_loader = prepare_dataloaders(args.data_dir, n_gpu)
    
    if rank == 0:
        # get logger ready
        if args.log_dir != '':
            if not os.path.isdir(args.log_dir):
                os.makedirs(args.log_dir)
                os.chmod(args.log_dir, 0o775)
            logger = Tacotron2Logger(args.log_dir)

        # get ckpt_dir ready
        if args.ckpt_dir != '' and not os.path.isdir(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
            os.chmod(args.ckpt_dir, 0o775)

    #model.train()
    # ================ MAIN GTA LOOP! ===================
        for batch in train_loader:

            x, y = (model.module if n_gpu > 1 else model).parse_batch(batch)
            y_pred = model(x)
            mel_out, mel_out_postnet, gate_out, _ = y_pred
            save_mel((mel_out, mel_out_postnet), args.mel_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('-d', '--data_dir', type = str, default = 'data',
                        help = 'directory to load data')
    parser.add_argument('-cp', '--ckpt_pth', type = str, default = '',
                        help = 'path to load checkpoints')
    parser.add_argument('-mp', '--mel_path', type = str, default = '',
                        help = 'path to save mels')
    parser.add_argument('-mig', '--mig_id', type = str, default = '',
                        help = 'mig_id_cluster')

    args = parser.parse_args()
    if args.mig_id != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.mig_id
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    GTA(args)
