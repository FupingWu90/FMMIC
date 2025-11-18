# -*- coding:utf-8 -*-

# import copy
import torch
from torch.utils import tensorboard
import logging, os
from  torch.optim.lr_scheduler import _LRScheduler, ExponentialLR, CosineAnnealingLR
from evaluation import MedMNIST_TEST_Tasks
import torch.nn.functional as F
import torch.nn as nn
import datetime
import numpy as np
import matplotlib.pyplot as plt
from timm.scheduler import CosineLRScheduler
from peft import LoraConfig, get_peft_model,inject_adapter_in_model, TaskType


class Trainer(object):
    def __init__(self,model,checkpoint,config,train_loaders,val_loaders, test_loaders,save_dir):

        if 'lora' in config.backbone_update:
            # LoRA configuration
            target_modules = None
            if config.backbone in ['dino2_base']:
                target_modules = [
                    "qkv",  # Transformer blocks in the encoder part
                ]
            elif config.backbone in ['densenet121']:
                target_modules = []
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Conv2d) and 'features' in name and 'conv' in name:
                        target_modules.append(name)

            lora_config = LoraConfig(
                r=8,  # Rank of LoRA adaptation matrices (adjust this parameter)
                lora_alpha=16,  # Scaling factor
                lora_dropout=0.1,  # Dropout rate for LoRA
                bias="none",  # Whether to add bias
                task_type=TaskType.FEATURE_EXTRACTION, #"FEATURE_EXTRACTION",  # Classification task
                target_modules=target_modules
            )

            # Wrap the model with LoRA
            model = inject_adapter_in_model(lora_config,model)

        self.model = model.cuda()
        self.config = config



        ##  optimizer
        if 'sft' in config.backbone_update:

            if 'lora' in config.backbone_update:

                optimizer_grouped_parameters = []
                for name,param in self.model.named_parameters():
                    if 'lora' in name:
                        optimizer_grouped_parameters.append(param)
                    else:
                        param.requires_grad = False
                for param in self.model.get_classifier().parameters():
                    param.requires_grad = True
                    optimizer_grouped_parameters.append(param)

                self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.bb_lr, weight_decay=1e-2)

            else:

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
                    self.encoder_optimizer = torch.optim.AdamW([{'params': other_params}], lr=config.bb_lr,weight_decay=1e-2)
                    self.decoder_optimizer = torch.optim.AdamW([{'params': classifier_params}], lr=config.lr,
                                                               weight_decay=1e-2)
                    # print(model.get_nonbackbon_params())

        elif 'scratch' in config.backbone_update:
            self.optimizer = torch.optim.AdamW([{'params': self.model.parameters()}], lr=config.lr,
                                               weight_decay=1e-2)
        else:
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
        self.best_iter = 0
        self.best_metric = 0.0001
        self.iter_counter = 0
        if checkpoint is not None:
            self.start_epoch = checkpoint['epoch']
            self.best_metric = checkpoint['best_metric']
            self.best_iter = checkpoint['best_iter']
            self.iter_counter = checkpoint['iter_counter']

            if 'sft' in self.config.backbone_update:
                if 'lora' in self.config.backbone_update:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                else:
                    self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
                    self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])

            else:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.model.load_state_dict(checkpoint['state_dict'])

        if 'sft' in self.config.backbone_update:
            if 'lora' in self.config.backbone_update:
                self.scheduler = ExponentialLR(optimizer=self.optimizer, gamma=config.lr_decay)
            else:
                self.scheduler = ExponentialLR(optimizer=self.decoder_optimizer, gamma=config.lr_decay)

        else:
            self.scheduler = ExponentialLR(optimizer=self.optimizer, gamma=config.lr_decay) #CosineLRScheduler(self.optimizer, t_initial=self.config.total_epoch, cycle_limit=1, t_in_epochs=True) #ExponentialLR(optimizer=self.optimizer, gamma=config.lr_decay)

        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.test_loaders = test_loaders


        #self.epochs = config.epochs
        self.save_dir = save_dir
        self.logger = self._create_logger(save_dir)



        self.writer = tensorboard.SummaryWriter(os.path.join(save_dir, 'tensorboard_writer'))
        self.wrt_step = 0
        self.threshold = 0.9

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
        eval_metrics = []
        loss_file = self.save_dir + '/loss_change.npy'
        metrics_file = self.save_dir + '/eval_metrics.npy'
        if os.path.exists(loss_file):
            loss_list = np.load(loss_file, allow_pickle=True).tolist()
        if os.path.exists(metrics_file):
            eval_metrics = np.load(metrics_file, allow_pickle=True).tolist()

        # ii = 0
        if 'lp' in self.config.backbone_update:
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.get_classifier().parameters():
                param.requires_grad = True

        ## total params
        total_params = sum(p.numel() for p in self.model.parameters())
        ## update params
        unfreeze_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        update_ratio_params = float(unfreeze_params) / total_params
        self.logger.info(
            f'\n-------------------------unfreezed params ratio:  {unfreeze_params} / {total_params} = {update_ratio_params}  -------------------------- \n')


        for epoch in range(self.start_epoch,self.config.total_epoch):
            self.model.train()

            dataloader = iter(self.train_loaders)
            train_starttime = datetime.datetime.now()


            for batch_idx in range(self.config.iter_per_epoch):  # self.outer_iter_per_epoch
                if self.iter_counter>self.config.total_Iter_Num-1:
                    break

                if self.config.backbone_update in ['sft','semi_sft']:
                    main_task_lr = self.decoder_optimizer.param_groups[0]['lr']

                else:
                    main_task_lr = self.optimizer.param_groups[0]['lr']

                if 'semi' in self.config.backbone_update:
                    img_l, label_l,img_u = next(dataloader)
                    img_l = img_l.cuda()
                    img_u = img_u.cuda()
                else:
                    img_l, label_l = next(dataloader)
                    img_l = img_l.cuda()
                if self.main_type == 'binary':
                    gt = label_l.float().cuda()
                elif self.main_type == "multi-label-binary-class":
                    gt = label_l.float().cuda()
                else:
                    gt = label_l.cuda()

                # update params
                self.model.zero_grad()

                # with autocast():
                task_out = self.model(img_l)
                total_loss = self.model_fit(task_out, gt)

                if 'semi' in self.config.backbone_update:
                    # Function to generate pseudo-labels
                    # def get_pseudo_labels(model, unlabeled_data, threshold=0.9):
                    self.model.eval()
                    with torch.no_grad():
                        outputs_u = self.model(img_u)

                        if self.config.dataname == 'ChestMNIST':
                            sigm = nn.Sigmoid()
                            task_predict = sigm(outputs_u)
                            max_probs, pseudo_labels = task_predict, (task_predict>0.5)*1

                            mask = max_probs > self.threshold
                        else:
                            if self.config.num_cls == 2:
                                sigm = nn.Sigmoid()
                                task_predict = sigm(outputs_u.squeeze(1))
                                max_probs, pseudo_labels = task_predict, (task_predict > 0.5) * 1
                                mask = (max_probs > self.threshold) | (max_probs < 1 - self.threshold)
                            else:
                                task_predict = outputs_u.softmax(dim=-1)
                                max_probs, pseudo_labels = torch.max(task_predict, dim=1)
                                mask = max_probs > self.threshold

                        # Apply thresholding

                        filtered_pseudo_labels, filtered_unlabeled_data =  pseudo_labels[mask], img_u[mask]

                    self.model.train()
                    if len(filtered_unlabeled_data) > 0:
                        unlabeled_outputs = self.model(filtered_unlabeled_data)
                        unlabeled_loss = self.model_fit(unlabeled_outputs, filtered_pseudo_labels)

                        if self.iter_counter < 1000:
                            balance_param = 0.0
                        elif self.iter_counter < 2000:
                            balance_param = (self.iter_counter - 1000) / (2000 - 1000)
                        else:
                            balance_param = 1.0

                        total_loss += balance_param * unlabeled_loss


                loss_list.append(total_loss.item())
                total_loss.backward()

                # log
                if batch_idx % 10 == 0:
                    self.logger.info(
                        f'\n epoch:{epoch}, iter: {batch_idx}, main task lr: {main_task_lr},  task loss: {total_loss}')

                if self.config.backbone_update in ['sft','semi_sft']:
                    self.encoder_optimizer.step()
                    self.decoder_optimizer.step()

                else:
                    self.optimizer.step()

                num_evl = 10
                if self.config.total_Iter_Num < num_evl:
                    num_evl = self.config.total_Iter_Num

                if (self.iter_counter % (self.config.total_Iter_Num // num_evl) == 0) or (self.iter_counter == self.config.total_Iter_Num - 1):
                    acc_metric = MedMNIST_TEST_Tasks(self.val_loaders, self.model, self.save_dir, 'val')

                    acc1, auc1 = acc_metric
                    eval_metrics.append([acc1, auc1])

                    if acc1 > self.best_metric:
                        self.best_metric = acc1
                        self.best_iter = self.iter_counter
                        # best_model_file = os.path.join(self.save_dir, 'model_param.pkl')
                        # if os.path.exists(best_model_file):
                        #     os.remove(os.path.join(self.save_dir, 'model_param.pkl'))
                        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model_param.pkl'))

                    if self.config.backbone_update in ['sft', 'semi_sft']:
                        torch.save({'epoch': epoch + 1,
                                    'state_dict': self.model.state_dict(),
                                    'best_iter': self.best_iter,
                                    'best_metric': self.best_metric,
                                    'iter_counter': self.iter_counter,
                                    'encoder_optimizer': self.encoder_optimizer.state_dict(),
                                    'decoder_optimizer': self.decoder_optimizer.state_dict()},
                                   self.save_dir + 'checkpoint.pth.tar')
                        np.save(self.save_dir + '/eval_metrics.npy', np.array(eval_metrics))
                        np.save(self.save_dir + '/loss_change.npy', np.array(loss_list))

                    else:
                        torch.save({'epoch': epoch + 1,
                                    'state_dict': self.model.state_dict(),
                                    'best_iter': self.best_iter,
                                    'best_metric': self.best_metric,
                                    'iter_counter': self.iter_counter,
                                    'optimizer': self.optimizer.state_dict()}, self.save_dir + 'checkpoint.pth.tar')
                        np.save(self.save_dir + '/eval_metrics.npy', np.array(eval_metrics))
                        np.save(self.save_dir + '/loss_change.npy', np.array(loss_list))

                self.iter_counter += 1

                if self.iter_counter % self.config.lr_decay_iters == 0:
                    self.scheduler.step()

                if self.iter_counter == self.config.total_Iter_Num:
                    break


            train_endtime = datetime.datetime.now()  #MedMNIST_TEST_Tasks
            self.logger.info(f'\n epoch: {epoch},  train time: {(train_endtime - train_starttime).seconds}')


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
        if len(eval_metrics) != 0:
            np.save(self.save_dir + '/eval_metrics.npy', np.array(eval_metrics))
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

        torch.save({'best_iter': self.best_iter,
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
            if x_output.dim()==2:
                x_output = x_output.squeeze(1)
            loss = F.cross_entropy(x_pred, x_output)
        elif self.main_type == 'binary':
            x_output = x_output.float().cuda()
            if x_output.dim()==2:
                x_output = x_output.squeeze(1)
            loss = F.binary_cross_entropy_with_logits(x_pred.squeeze(1), x_output)
        elif self.main_type == 'multi-label-binary-class':
            loss = F.binary_cross_entropy_with_logits(x_pred, x_output)

        return loss








