import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from os.path import join as pjoin

from models.length_estimator.mlp import CLIPAdapterLengthRegressor
from models.length_estimator.mlp_trainer import LengthEstimatorTrainer

from options.train_option import TrainLenEstOptions

from utils.plot_script import plot_3d_motion
from utils.motion_process import recover_from_ric
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from utils.paramUtil import t2m_kinematic_chain, kit_kinematic_chain

from data.t2m_dataset import Text2MotionLenDataset
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper

def load_len_estimator(opt):
    model = CLIPAdapterLengthRegressor(
        clip_dim=512, # opt.clip_dim,
        adapter_dim=256, # opt.adapter_dim,
        hidden_dim=512 # opt.hidden_dim
    )
    return model

if __name__ == '__main__':
    parser = TrainLenEstOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    model = load_len_estimator(opt)

    clip_version = 'ViT-B/32'
    if opt.dataset_name == 't2m':
        opt.data_root = './dataset/HumanML3D/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.dim_pose = 263
        opt.max_motion_length = 196
        opt.max_motion_frame = 196
        opt.max_motion_token = 55
    else:
        NotImplementedError(f"Invalid dataset name {opt.dataset_name}")

    mean = np.load('./dataset/HumanML3D/mean.npy')
    std = np.load('./dataset/HumanML3D/std.npy')
    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')

    train_dataset = Text2MotionLenDataset(opt, mean, std, train_split_file)
    val_dataset = Text2MotionLenDataset(opt, mean, std, val_split_file)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, drop_last=True)

    trainer = LengthEstimatorTrainer(opt, model)
    trainer.train(train_loader, val_loader)