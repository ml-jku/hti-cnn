import argparse
import os
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tensorboardX import SummaryWriter

from pyll.base import invoke_dataset_from_config, invoke_model_from_config, TorchModel
from pyll.config import Config
from pyll.utils.misc import extract_named_args, try_to_number_or_bool
from pyll.utils.workspace import Workspace


class PyLL(object):
    def __init__(self, enable_workspace=True, enable_optimizer=True):
        self.best_performance = 0
        self.enable_workspace = enable_workspace
        self.enable_optimizer = enable_optimizer
        args, unknown_args = self.__parse_args__()
        # --
        if args.gpu != -1:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
            self.cuda_is_available = torch.cuda.is_available()
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            self.cuda_is_available = False

        # Read Config
        config = Config()
        for k, v in vars(args).items():
            if isinstance(v, str) and (v.lower() == "false" or v.lower() == "true"):
                v = (v.lower() == "true")
            config.override(k, v)
            # Init Dataset
        datasets = invoke_dataset_from_config(config)
        # Init Model        
        model: TorchModel = invoke_model_from_config(config, datasets[list(datasets.keys())[0]])
        loss = model.loss
        model = self.cuda(torch.nn.DataParallel(model))
        print("Trainable Parameters: {:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

        # Optimizer
        if enable_optimizer:
            self.optimizer = config.get_value("optimizer")(params=model.parameters(), **config.get_value("optimizer_params"))

        # optionally resume from a checkpoint
        if args.checkpoint:
            if os.path.isfile(args.checkpoint):
                if enable_workspace:
                    self.workspace = Workspace(resume=args.checkpoint)
                    print("Workspace: {}".format(self.workspace.workspace_dir))
                print("=> loading checkpoint '{}'".format(args.checkpoint))
                checkpoint = torch.load(args.checkpoint)

                if enable_optimizer:
                    # check if optimizer params have been specified on the command line
                    override = extract_named_args(unknown_args)
                    optimizer_params = {}
                    for k, v in override.items():
                        name = k[2:] if "--" in k else k  # remove leading --
                        if name.startswith("optimizer_params."):
                            name = name[len("optimizer_params."):]
                            value = v if v.startswith('"') or v.startswith("'") else try_to_number_or_bool(v)
                            optimizer_params[name] = value
                    if len(optimizer_params) > 0:
                        print("=> overriding optimizer params from command line [{}]".format(optimizer_params))
                        for k, v in checkpoint['optimizer']['param_groups'][0].items():
                            if k in optimizer_params:
                                checkpoint['optimizer']['param_groups'][0][k] = optimizer_params[k]

                # assign values from checkpoint
                self.samples_seen = checkpoint['samples_seen'] if "samples_seen" in checkpoint else 0
                self.epoch = checkpoint['epoch']
                self.best_performance = checkpoint['performance']
                try:
                    model.load_state_dict(checkpoint['state_dict'])
                except RuntimeError:
                    # if state dict is missing values try initializing partial state
                    state = model.state_dict()
                    state.update(checkpoint['state_dict'])
                    model.load_state_dict(state)

                if enable_optimizer:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.checkpoint))
                exit(1)
        else:
            # Init Workspace
            if enable_workspace:
                self.workspace = Workspace(config.get_value("workspace", "."), config.get_value("name", "run"), comment=config.get_value("comment", "None"))
                print("Workspace: {}".format(self.workspace.workspace_dir))
            self.epoch, self.samples_seen = 0, 0

        if enable_workspace:
            # Init SummaryWriters
            summaries = {}
            for dataset, _ in datasets.items():
                summaries[dataset] = SummaryWriter(os.path.join(self.workspace.statistics_dir, dataset))
            self.summaries = summaries

        cudnn.benchmark = True

        # set properties
        self.model = model
        self.loss = loss
        self.datasets = datasets
        self.config = config

    @staticmethod
    def __parse_args__():
        parser = argparse.ArgumentParser(description='PyLL Training')

        parser.add_argument('--config', '-c', metavar='CONFIG', default='config.json',
                            help='run config')
        parser.add_argument('-g', '--gpu', default="0", type=str, metavar='N',
                            help='gpu ids for training (default: 0); set to -1 to disable GPU support')
        parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
        parser.add_argument('--print-freq', '-p', default=10, type=int,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                            help='path to checkpoint (default: none)')
        parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                            help='evaluate model on validation set')
        parser.add_argument('-i', '--inference', type=str, metavar="DSET", default=None,
                            help='perform inference on specified dataset')
        args, unknown_args = parser.parse_known_args()
        return args, unknown_args

    def train_step(self, input, target, epoch):
        if not self.enable_optimizer:
            print("Unable to train without optimizer")
            return

        if self.config.has_value("lr_schedule"):
            self.adjust_learning_rate(epoch)

        # compute output
        prediction = self.model(input)
        loss = self.loss(prediction, target)
        # regularization
        if self.config.has_value("regularization"):
            regularization = self.config.get_value("regularization", None)
            l1_lambda = regularization.get_value("l1", 0)
            l2_lambda = regularization.get_value("l2", 0)
            l1 = self.cuda(torch.tensor(0, requires_grad=True).float())
            l2 = self.cuda(torch.tensor(0, requires_grad=True).float())
            if l1_lambda > 0:
                for name, W in self.model.named_parameters():
                    if "bias" not in name:
                        l1 = l1 + W.abs().sum()
                loss += l1_lambda * l1
            if l2_lambda > 0:
                for name, W in self.model.named_parameters():
                    if "bias" not in name:
                        l2 = l2 + torch.mul(W, W).sum()
                loss += l2_lambda * l2

        # backprop
        self.optimizer.zero_grad()
        loss.backward()

        if self.config.has_value("clip_grad_norm") and self.config.clip_grad_norm.has_value("max_norm"):
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm.max_norm,
                                     norm_type=self.config.clip_grad_norm.get_value("norm_type", 2))

        if self.config.has_value("clip_grad_value") and self.config.clip_grad_value.has_value("clip_value"):
            nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.config.clip_grad_value.clip_value)

        self.optimizer.step()
        # update iteration counter
        self.samples_seen += input.size(0)  # add number of samples
        self.epoch = epoch
        # --
        return loss, prediction

    def save_checkpoint(self, performance, is_best, filename='checkpoint.pth.tar', model_best_filename='model_best.pth.tar'):
        if not self.enable_workspace:
            print("Unable to save checkpoint without workspace")
            return

        state = {
            'samples_seen': self.samples_seen,
            'epoch': self.epoch,
            'model': self.config.get_value("model", "unknown"),
            'state_dict': self.model.state_dict(),
            'performance': performance,
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, os.path.join(self.workspace.checkpoint_dir, filename))
        if is_best:
            shutil.copyfile(os.path.join(self.workspace.checkpoint_dir, filename), os.path.join(self.workspace.checkpoint_dir, model_best_filename))

    def adjust_learning_rate(self, current_epoch):
        if not self.enable_optimizer:
            return

        """Sets the learning rate to the initial LR decayed by 10% every 30 epochs"""
        lr = self.config.optimizer_params.lr
        decay_rate = self.config.lr_schedule["decay_rate"]
        decay_epoch = self.config.lr_schedule["decay_epoch"]
        lr = lr * (decay_rate ** (current_epoch // decay_epoch))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def cuda(self, var):
        if self.cuda_is_available:
            return var.cuda()
        return var
