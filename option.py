import argparse
import torch
import numpy as np
import random
import distutils
import os

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def str2bool(self, v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def initialize(self):
        # system
        self.parser.add_argument(
            '--name', type=str, default='PSGN_5000_2_1', help='logger & saver name')
        self.parser.add_argument('--verbose', type=self.str2bool, default=True, help='show debug information')
        self.parser.add_argument('--use_cuda', type=self.str2bool, default=True, help='use cuda or not')
        self.parser.add_argument('--gpu_ids',  type=int, default=[0], help='gpu device id for parallel', nargs='+')
        # dataset
        self.parser.add_argument('--data_type', type=str, default='shapenet', help='dataset name')
        self.parser.add_argument('--data_root', type=str, default='/mnt/sdd/weicai/dataset/ShapeNetCore.v2', help='dataset root directory')
        self.parser.add_argument('--use_diff_sub', type=self.str2bool, default=False, help='use differentiable subdivison')
        self.parser.add_argument('--use_z_weight', type=self.str2bool, default=False, help='use z weight in projection')
        self.parser.add_argument('--use_symm_edge_aggr', type=self.str2bool, default=False, help='add symmetric edges in graph aggregation')
        self.parser.add_argument('--use_symm_edge_update', type=self.str2bool, default=False, help='add symmetric edges in graph update')
        self.parser.add_argument('--use_offset', type=self.str2bool, default=False, help='learn coord offset')
        self.parser.add_argument('--form_batch', type=self.str2bool, default=False, help='form batch in local block')
        self.parser.add_argument('--train_category', type=str, default='plane', help='specify categoty during training', nargs='+')
        self.parser.add_argument('--eval_category',  type=str, default='plane', help='specify categoty during evaluating', nargs='+')
        self.parser.add_argument('--batch_size', type=int, default=32, help='batch size')  #32
        self.parser.add_argument('--num_workers', type=int, default=8, help='data loader num process')
        self.parser.add_argument('--pin_mem', type=self.str2bool, default=False, help='use pin memory in dataloader')
        # hypervalue
        self.parser.add_argument('--seed', type=int, default=1024, help='random seed')
        self.parser.add_argument('--epoch', type=int, default=100, help='training epoch num')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        self.parser.add_argument('--weight_decay', type=float, default=5e-6, help='layer L2 loss weight decay') 
        self.parser.add_argument('--obj_file', type=str, default='./dataset/p2m/unit_ball.obj', help='predefined mesh')
        # checking
        self.parser.add_argument('--use_logger', type=self.str2bool, default=True, help='check intermediate result')
        self.parser.add_argument('--save_path', type=str, default='./save', help='checkpoint path')
        self.parser.add_argument('--log_path', type=str, default='./log', help='training log path')
        self.parser.add_argument('--loss_freq_iter', type=int, default=20, help='record loss every n iteration')
        self.parser.add_argument('--train_mesh_freq_iter', type=int, default=2000, help='record mesh every n iteration during training')
        self.parser.add_argument('--eval_mesh_freq_iter', type=int, default=50, help='record mesh every n iteration during evaluation')
        self.parser.add_argument('--eval_freq_epoch', type=int, default=1, help='evaluation every n epoch')
        self.parser.add_argument('--check_freq_iter', type=int, default=500, help='check grad and projection every n iteration')
        # loss weight
        self.parser.add_argument('--use_sample', type=self.str2bool, default=False)
        self.parser.add_argument('--point_loss_weight', type=float, default=1.0)
        self.parser.add_argument('--edge_loss_weight', type=float, default=0.3)
        self.parser.add_argument('--norm_loss_weight', type=float, default=1.6e-4)
        self.parser.add_argument('--laplace_loss_weight', type=float, default=0.5)
        self.parser.add_argument('--move_loss_weight', type=float, default=0.1)
        self.parser.add_argument('--convex_loss_weight', type=float, default=0.01)
        self.parser.add_argument('--symm_loss_weight', type=float, default=0.1)
        self.parser.add_argument('--img_loss_weight', type=float, default=1.0)
        self.parser.add_argument('--use_new_laploss', type=self.str2bool, default=False)
        # layer info
        self.parser.add_argument('--hidden_channel', type=int, default=256)
        self.parser.add_argument('--block_num', type=int, default=6)

        self.parser.add_argument('--vis', type=int, default=1)  # visualize

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt, unparsed = self.parser.parse_known_args()

        if self.opt.seed is not None:
            torch.manual_seed(self.opt.seed)
            np.random.seed(self.opt.seed)
            random.seed(self.opt.seed)
        
        self.opt.muti_gpu = False
        if len(self.opt.gpu_ids) > 1:
            self.opt.muti_gpu = True

        if self.opt.verbose:
            print('+ Config Parameters')
            log_dir = self.opt.log_path
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            parameter_file = open(log_dir + "/train_param.txt", 'w')
            for key, value in self.opt.__dict__.items():
                print('  - {:25}: {}'.format(key, value))
                parameter_file.write(str(key) + " " + str(value) + "\n")

        self.opt.weight_dict = {}
        self.opt.weight_dict['point_loss_weight'] = self.opt.point_loss_weight
        self.opt.weight_dict['edge_loss_weight'] = self.opt.edge_loss_weight
        self.opt.weight_dict['norm_loss_weight'] = self.opt.norm_loss_weight
        self.opt.weight_dict['laplace_loss_weight'] = self.opt.laplace_loss_weight
        self.opt.weight_dict['move_loss_weight'] = self.opt.move_loss_weight
        self.opt.weight_dict['convex_loss_weight'] = self.opt.convex_loss_weight
        self.opt.weight_dict['symm_loss_weight'] = self.opt.symm_loss_weight
        self.opt.weight_dict['img_loss_weight'] = self.opt.img_loss_weight

        return self.opt
        
def create_parser():
    return Options().parse()
