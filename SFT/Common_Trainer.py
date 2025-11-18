# -*- coding:utf-8 -*-

import torch
from torch.utils import tensorboard
import logging, os
from  torch.optim.lr_scheduler import _LRScheduler, ExponentialLR, CosineAnnealingLR
from evaluation import MedMNIST_TEST_Tasks
import torch.nn.functional as F
import datetime
import numpy as np
import matplotlib.pyplot as plt
from timm.scheduler import CosineLRScheduler


class Trainer(object):
    def __init__(self,model,checkpoint,config,train_loaders,val_loaders, test_loaders,save_dir):
        self.model = model.cuda()
        self.config = config

        ##  optimizer
        if config.backbone_update in ['ft',]:
            classifier_params = self.model.get_classifier().parameters()
            classifier_param_ids = set(id(param) for param in classifier_params)

            other_params = []
            for name, param in self.model.named_parameters():
                if id(param) not in classifier_param_ids:
                    other_params.append(param)

            if config.Optim == 'SGD':
                self.encoder_optimizer = torch.optim.SGD([{'params': other_params}], lr=config.bb_lr,
                                                         weight_decay=5e-4, momentum=config.momentum)
                self.decoder_optimizer = torch.optim.SGD([{'params': classifier_params}], lr=config.lr,
                                                         weight_decay=5e-4, momentum=config.momentum)

            elif config.Optim == 'Adam':
                self.encoder_optimizer = torch.optim.Adam([{'params': other_params}], lr=config.bb_lr,
                                                          weight_decay=5e-4)
                self.decoder_optimizer = torch.optim.Adam([{'params': classifier_params}], lr=config.lr,
                                                          weight_decay=5e-4)

            elif config.Optim == 'AdamW':
                self.encoder_optimizer = torch.optim.AdamW([{'params': other_params}], lr=config.bb_lr)
                self.decoder_optimizer = torch.optim.AdamW([{'params': classifier_params}], lr=config.lr,
                                                           weight_decay=1e-2)
                # print(model.get_nonbackbon_params())

        elif config.backbone_update == 'fz':
            classifier_params = self.model.get_classifier().parameters()

            if config.Optim == 'SGD':
                self.optimizer = torch.optim.SGD([{'params': classifier_params}], lr=config.lr,
                                                 weight_decay=5e-4, momentum=config.momentum)

            elif config.Optim == 'Adam':
                self.optimizer = torch.optim.Adam([{'params': classifier_params}], lr=config.lr,
                                                  weight_decay=5e-4)
            elif config.Optim == 'AdamW':
                self.optimizer = torch.optim.AdamW([{'params': classifier_params}], lr=config.lr,
                                                   weight_decay=1e-2)




        self.start_epoch = 0
        self.best_epoch = 0
        self.best_metric = 0.0001
        self.iter_counter = 0
        if checkpoint is not None:
            self.start_epoch = checkpoint['epoch']
            self.best_metric = checkpoint['best_metric']
            self.best_epoch = checkpoint['best_epoch']
            self.iter_counter = checkpoint['iter_counter']

            if self.config.backbone_update in ['ft',]:
                self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
                self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])

            elif self.config.backbone_update in ['fz',]:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.model.load_state_dict(checkpoint['state_dict'])

        if self.config.backbone_update in ['ft',]:
            self.scheduler = ExponentialLR(optimizer=self.decoder_optimizer, gamma=config.lr_decay)

        elif self.config.backbone_update in ['fz',]:
            self.scheduler = ExponentialLR(optimizer=self.optimizer, gamma=config.lr_decay) #CosineLRScheduler(self.optimizer, t_initial=self.config.total_epoch, cycle_limit=1, t_in_epochs=True) #ExponentialLR(optimizer=self.optimizer, gamma=config.lr_decay)

        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.test_loaders = test_loaders


        #self.epochs = config.epochs
        self.save_dir = save_dir
        self.logger = self._create_logger(save_dir)
        self.writer = tensorboard.SummaryWriter(os.path.join(save_dir, 'tensorboard_writer'))
        self.wrt_step = 0

        if self.config.dataname == 'ChestMNIST':
            self.main_type = "multi-label-binary-class"
        else:
            if self.config.num_cls == 2:
                self.main_type = 'binary'
            else:
                self.main_type = 'multi_cls'

        #self.test_dataset_path = config.datapath_test


    def train(self,):


        self.logger.info(f'\n-------------------------Start training: total {self.config.total_epoch} epoches, {self.config.iter_per_epoch} iterations per epoch  -------------------------- ')
        # train

        loss_list = []

        # ii = 0
        if self.config.backbone_update == 'fz':
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.get_classifier().parameters():
                param.requires_grad = True


        for epoch in range(self.start_epoch,self.config.total_epoch):
            self.model.train()

            dataloader = iter(self.train_loaders)
            train_starttime = datetime.datetime.now()


            for batch_idx in range(self.config.iter_per_epoch):  # self.outer_iter_per_epoch

                if self.config.backbone_update in ['ft',]:
                    main_task_lr = self.decoder_optimizer.param_groups[0]['lr']

                elif self.config.backbone_update in ['fz',]:
                    main_task_lr = self.optimizer.param_groups[0]['lr']

                img, label = next(dataloader)
                img = img.cuda()
                if self.main_type == 'binary':
                    gt = label.float().cuda()
                elif self.main_type == "multi-label-binary-class":
                    gt = label.float().cuda()
                else:
                    gt = label.cuda()

                # update params
                self.model.zero_grad()

                # with autocast():
                task_out = self.model(img)
                task_loss = self.model_fit(task_out, gt)


                loss_list.append(task_loss.item())
                task_loss.backward()

                # log
                if batch_idx % 10 == 0:
                    self.logger.info(
                        f'\n epoch:{epoch}, iter: {batch_idx}, main task lr: {main_task_lr},  task loss: {task_loss}')

                if self.config.backbone_update in ['ft',]:
                    self.encoder_optimizer.step()
                    self.decoder_optimizer.step()

                elif self.config.backbone_update in ['fz',]:
                    self.optimizer.step()

                self.iter_counter += 1

                if self.iter_counter % self.config.lr_decay_iters == 0:
                    self.scheduler.step()
            # self.scheduler.step(epoch=epoch)


            train_endtime = datetime.datetime.now()  #MedMNIST_TEST_Tasks
            self.logger.info(f'\n epoch: {epoch},  train time: {(train_endtime - train_starttime).seconds}')

            num_evl = 10
            if self.config.total_epoch <num_evl:
                num_evl = self.config.total_epoch

            if (epoch % (self.config.total_epoch//num_evl) ==0) or (epoch==self.config.total_epoch-1):
                acc_metric = MedMNIST_TEST_Tasks(self.val_loaders, self.model, self.save_dir, 'val')

                acc1, auc1 = acc_metric

                if acc1 > self.best_metric:
                    self.best_metric = acc1
                    self.best_epoch = epoch

                    torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model_param.pkl'))

            if self.config.backbone_update in ['ft',]:
                torch.save({'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'best_epoch': self.best_epoch,
                            'best_metric': self.best_metric,
                            'iter_counter': self.iter_counter,
                            'encoder_optimizer': self.encoder_optimizer.state_dict(),
                            'decoder_optimizer': self.decoder_optimizer.state_dict()},
                           self.save_dir + 'checkpoint.pth.tar')

            elif self.config.backbone_update in ['fz',]:
                torch.save({'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'best_epoch': self.best_epoch,
                            'best_metric': self.best_metric,
                            'iter_counter': self.iter_counter,
                            'optimizer': self.optimizer.state_dict()}, self.save_dir + 'checkpoint.pth.tar')

        self.logger.info(f'\n-------------------------End training -------------------------- ')

        # evaluate
        self.logger.info(f'\n-------------------------Start Testing -------------------------- ')

        self.model.load_state_dict(torch.load(os.path.join(self.save_dir, 'model_param.pkl'), weights_only=True))
        acc_metric = MedMNIST_TEST_Tasks(self.test_loaders, self.model,self.save_dir, 'test')

        acc1, auc1 = acc_metric
        np.save(self.save_dir + 'test_ndarray.npy', np.array([[acc1, auc1]]))
        # save model
        # torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model_param.pkl'))

        # save loss curve
        if len(loss_list) != 0:
            np.save(self.save_dir + '/loss_change.npy', np.array(loss_list))
            # %% plot loss

            plt.plot(loss_list)
            plt.title("Training Loss")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.savefig(self.save_dir + "/train_loss.png")
            plt.close()


        self.logger.info(f'\n-------------------------End Testing -------------------------- ')

        torch.save({'best_epoch': self.best_epoch,
                    'best_metric': self.best_metric,
                    }, self.save_dir + 'checkpoint_finish.pth.tar')

        checkpoint_file = self.save_dir + 'checkpoint.pth.tar'
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

        del self.logger


    def _create_logger(self, save_dir):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logfile = os.path.join(save_dir, 'training.log')

        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(logging.DEBUG)

        logger.addHandler(fh)

        return logger




    def model_fit(self, x_pred, x_output):
        if self.main_type == 'multi_cls':
            loss = F.cross_entropy(x_pred, x_output.squeeze(1))
        elif self.main_type == 'binary':
            loss = F.binary_cross_entropy_with_logits(x_pred.squeeze(1), x_output.squeeze(1))
        elif self.main_type == 'multi-label-binary-class':
            loss = F.binary_cross_entropy_with_logits(x_pred, x_output)

        return loss









