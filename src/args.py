import os
import yaml
import argparse
import numpy as np
import torch.utils
import torchvision.datasets as dset

from copy import copy
from model import utils
import dataset
from dataset import init


class Parser(object):
    def __init__(self):
        parser = argparse.ArgumentParser("DARTS-")

        # general options
        parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
        parser.add_argument('--space', type=str, default='s1', help='space index')
        parser.add_argument('--dataset', type=str, default='animal-skeleton', help='dataset')
        parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
        parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
        parser.add_argument('--seed', type=int, default=20, help='random seed')
        parser.add_argument('--resume', action='store_true', default=False, help='resume search')
        parser.add_argument('--debug', action='store_true', default=False, help='use one-step unrolled validation loss')
        parser.add_argument('--job_id', type=int, default=1, help='SLURM_ARRAY_JOB_ID number')
        parser.add_argument('--task_id', type=int, default=1, help='SLURM_ARRAY_TASK_ID number')

        # training options
        parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
        parser.add_argument('--batch_size', type=int, default=128, help='batch size')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
        parser.add_argument('--learning_rate_min', type=float, default=0.00001, help='min learning rate')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
        parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
        parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
        parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
        parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
        parser.add_argument('--unrolled', action='store_true', default=False,
                            help='use one-step unrolled validation loss')

        # one-shot model options
        parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
        parser.add_argument('--layers', type=int, default=3, help='total number of layers')
        parser.add_argument('--nodes', type=int, default=3, help='number of intermediate nodes per cell')

        # augmentation options
        parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
        parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
        parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
        parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')

        # logging options
        parser.add_argument('--save', type=str, default='experiments/search_logs', help='experiment name')
        parser.add_argument('--results_file_arch', type=str, default='results_arch',
                            help='filename where to write architectures')
        parser.add_argument('--results_file_perf', type=str, default='results_perf',
                            help='filename where to write val errors')
        parser.add_argument('--report_freq', type=float, default=1, help='report frequency')
        parser.add_argument('--report_freq_hessian', type=float, default=1, help='report frequency hessian')

        # early stopping
        parser.add_argument('--early_stop', type=int, default=2, choices=[0, 1, 2, 3],
                            help='early stop DARTS based on dominant eigenvalue. 0: no 1: yes 2: simulate 3: adaptive regularization')
        parser.add_argument('--window', type=int, default=5, help='window size of the local average')
        parser.add_argument('--es_start_epoch', type=int, default=10, help='when to start considering early stopping')
        parser.add_argument('--delta', type=int, default=4,
                            help='number of previous local averages to consider in early stopping')
        parser.add_argument('--factor', type=float, default=1.3, help='early stopping factor')
        parser.add_argument('--extra_rollback_epochs', type=int, default=0,
                            help='number of extra rollback epochs when deciding to increse regularization')
        parser.add_argument('--compute_hessian', action='store_true', default=False, help='compute or not Hessian')
        parser.add_argument('--max_weight_decay', type=float, default=243e-4, help='maximum weight decay')
        parser.add_argument('--mul_factor', type=float, default=3.0, help='multiplication factor')

        # randomNAS
        parser.add_argument('--eval_only', action='store_true', default=False, help='eval only')
        parser.add_argument('--randomnas_rounds', type=int, default=None,
                            help='number of evaluation rounds in RandomNAS')
        parser.add_argument('--n_samples', type=int, default=1000,
                            help='number of discrete architectures to sample during eval')

        # darts minus
        parser.add_argument('--auxiliary_skip', action='store_true', default=True, help='add an auxiliary skip')
        parser.add_argument('--auxiliary_operation', choices=['skip', 'conv1'], default='conv1',
                            help='specify auxiliary choices')
        parser.add_argument('--skip_beta', type=float, default=1.0,
                            help='ratio to overshoot or discount auxiliary skip')
        parser.add_argument('--decay', default='cosine', choices=[None, 'cosine', 'slow_cosine', 'linear'],
                            help='select scheduler decay on epochs')
        parser.add_argument('--decay_start_epoch', type=int, default=0, help='epoch to start decay')
        parser.add_argument('--decay_stop_epoch', type=int, default=300, help='epoch to stop decay')
        parser.add_argument('--decay_max_epoch', type=int, default=300, help='max epochs to decay')

        # Hessian ev calculation
        parser.add_argument('--ev_start_epoch', type=int, default=50,
                            help='starting id of checkpoint to load for ev calculation')
        parser.add_argument('--disable_cuda', action='store_true', default=False, help='disable cuda')

        # visualization
        parser.add_argument('--x', type=str, default='-1:1:301', help='A string with format xmin:x_max:xnum')
        parser.add_argument('--y', type=str, default='-1:1:301', help='A string with format ymin:y_max:ynum')
        parser.add_argument('--checkpoint_epoch', type=int, default=300, help='specify checkpoint epoch to load')
        parser.add_argument('--test_infer', action='store_true', default=False,
                            help='run inference to test whether the model is loaded correctly')
        parser.add_argument('--show', action='store_true', default=False, help='show graph before saving')
        parser.add_argument('--azim', type=float, default=-60, help='azimuthal angle for 3d landscape')
        parser.add_argument('--elev', type=float, default=30, help='elevation angle for 3d landscape')

        self.args = parser.parse_args()
        utils.print_args(self.args)


class Helper(Parser):
    def __init__(self):
        super(Helper, self).__init__()

        self.args._save = copy(self.args.save)
        self.args.save = './src/model/{}/{}/{}/{}_{}-{}'.format(self.args.save,
                                                    self.args.space,
                                                    self.args.dataset,
                                                    self.args.drop_path_prob,
                                                    self.args.weight_decay,
                                                    self.args.job_id)

        utils.create_exp_dir(self.args.save)

        config_filename = os.path.join(self.args._save, 'config.yaml')
        if not os.path.exists(config_filename):
            with open(config_filename, 'w') as f:
                yaml.dump(self.args_to_log, f, default_flow_style=False)

        self.args.n_classes = 5


        # set cutout to False if the drop_prob is 0
        if self.args.drop_path_prob == 0:
            self.args.cutout = False

    @property
    def config(self):
        return self.args

    @property
    def args_to_log(self):
        list_of_args = [
            "epochs",
            "batch_size",
            "learning_rate",
            "learning_rate_min",
            "momentum",
            "grad_clip",
            "train_portion",
            "arch_learning_rate",
            "arch_weight_decay",
            "unrolled",
            "init_channels",
            "layers",
            "nodes",
            "cutout_length",
            "report_freq_hessian",
            "early_stop",
            "window",
            "es_start_epoch",
            "delta",
            "factor",
            "extra_rollback_epochs",
            "compute_hessian",
            "mul_factor",
            "max_weight_decay",
        ]

        args_to_log = dict(filter(lambda x: x[0] in list_of_args,self.args.__dict__.items()))
        return args_to_log

    def get_train_val_loaders(self):
        dataset_name = 'animal-skeleton'
        dataset_args = {'train_batch_size': 64,
                        'eval_batch_size': 64,
                        'preprocess': False,
                        'path': 'F:/FLQ/DEEPLABCUTRELATED/skeleton/test',
                        'data_path': 'F: / FLQ / DEEPLABCUTRELATED / skeleton / test'}
        self.train_batch_size = dataset_args['train_batch_size']
        self.eval_batch_size = dataset_args['eval_batch_size']
        self.feeders, self.data_shape, self.num_class, self.A, self.parts = dataset.init.create(
            self.args.debug, self.args.dataset, **dataset_args
        )
        num_train = len(self.feeders['train'])
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))

        train_queue = torch.utils.data.DataLoader(
            self.feeders['train'],
            batch_size=self.args.batch_size,
            pin_memory=True,
            num_workers=0)

        valid_queue = torch.utils.data.DataLoader(
            self.feeders['eval'],
            batch_size=self.args.batch_size,
            pin_memory=True,
            num_workers=0)

        return train_queue, valid_queue, self.data_shape, self.num_class, self.A, self.parts


helper = Helper()
args = helper.config
#print(args)
beta_decay_scheduler = utils.DecayScheduler(base_lr=args.skip_beta,
                                            T_max=args.decay_max_epoch,
                                            T_start=args.decay_start_epoch,
                                            T_stop=args.decay_stop_epoch,
                                            decay_type=args.decay)