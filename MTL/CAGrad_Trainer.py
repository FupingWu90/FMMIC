# -*- coding:utf-8 -*-


import torch
import torch.nn as nn
from torch.utils import tensorboard
import logging, os
from torch.optim.lr_scheduler import _LRScheduler, ExponentialLR
from evaluation import Evaluate_UKBB_Index
import torch.nn.functional as F
import numpy as np
import time
from copy import deepcopy
from min_norm_solvers import MinNormSolver
from scipy.optimize import minimize, Bounds, minimize_scalar
from itertools import cycle
from dataset import *
import datetime
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model,inject_adapter_in_model, TaskType
# from cycler import cycler

Indice_Column_NumCls_Dict = {'Age':{'left_image_file_path':0,'right_image_file_path':6,'colum':0,'num_cls':1,'label_names':None,'metric':'mse','task':'regression'},
                             'Sex':{'left_image_file_path':0,'right_image_file_path':6,'colum':1,'num_cls':2,'label_names':['male','female'],'metric':'roc','task':'cls'},
                            'HbA1c':{'left_image_file_path':0,'right_image_file_path':6,'colum':2,'num_cls':1,'label_names':None,'metric':'mse','task':'regression'},
                            'glucose':{'left_image_file_path':0,'right_image_file_path':6,'colum':3,'num_cls':1,'label_names':None,'metric':'mse','task':'regression'},
                            'bmi':{'left_image_file_path':0,'right_image_file_path':6,'colum':4,'num_cls':1,'label_names':None,'metric':'mse','task':'regression'},
                            'icd':{'left_image_file_path':0,'right_image_file_path':6,'colum':5,'num_cls':2,'label_names':['negative','positive'],'metric':'roc','task':'cls'},
                            'ckd_binary':{'left_image_file_path':0,'right_image_file_path':6,'colum':6,'num_cls':2,'label_names':['negative','positive'],'metric':'roc','task':'cls'},
                            'egfr_6stage':{'left_image_file_path':0,'right_image_file_path':6,'colum':7,'num_cls':6,'label_names':['G1','G2','G3a','G3b','G4','G5'],'metric':'macro_f1','task':'cls'},
                            'ckd_6stage':{'left_image_file_path':0,'right_image_file_path':6,'colum':8,'num_cls':7,'label_names':['Normal','G1','G2','G3a','G3b','G4','G5'],'metric':'macro_f1','task':'cls'},
                            'diabete':{'left_image_file_path':0,'right_image_file_path':6,'colum':9,'num_cls':2,'label_names':['negative','positive'],'metric':'roc','task':'cls'},
                            'SBP':{'left_image_file_path':0,'right_image_file_path':6,'colum':10,'num_cls':1,'label_names':None,'metric':'mse','task':'regression'},

                             }
Task_List = ['Sex','icd', 'ckd_binary','diabete', 'Age', 'HbA1c', 'glucose', 'bmi','SBP']



class Trainer(object):
    def __init__(self,model,checkpoint,config,train_loaders,val_loaders, test_loaders,save_dir):
        if 'lora' in config.backbone_update:
            # LoRA configuration
            target_modules = None
            if config.backbone in ['dino2_base','retfound','dinov3']:
                target_modules = [
                    # "proj",  # Embedding layer, if you want to include it
                    "qkv",  # Transformer blocks in the encoder part
                    # "fc1",  # Layer normalization in the encoder part
                    # "fc2"
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
            model.backbone = inject_adapter_in_model(lora_config,model.backbone)

        self.model = model.cuda()
        self.config = config

        ##  optimizer
        if config.backbone_update in ['ft', ]:
            other_params = self.model.get_base_params()
            # classifier_param_names = set(name for name, _ in self.model.get_classifier().named_parameters())
            # other_params = [p for name, p in self.model.named_parameters() if name not in classifier_param_names]
            other_params_ids = set(id(param) for param in other_params)

            classifier_params = []
            for name, param in self.model.named_parameters():
                if id(param) not in other_params_ids:
                    classifier_params.append(param)

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
        elif config.backbone_update == 'lora':
            other_params = self.model.get_base_params()
            # classifier_param_names = set(name for name, _ in self.model.get_classifier().named_parameters())
            # other_params = [p for name, p in self.model.named_parameters() if name not in classifier_param_names]
            other_params_ids = set(id(param) for param in other_params)

            classifier_params = []
            for name, param in self.model.named_parameters():
                if id(param) not in other_params_ids:
                    classifier_params.append(param)

            optimizer_grouped_parameters = []
            for name, param in self.model.named_parameters():
                if 'lora' in name:
                    optimizer_grouped_parameters.append(param)
                else:
                    param.requires_grad = False
            for param in classifier_params:
                param.requires_grad = True
                optimizer_grouped_parameters.append(param)

            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.lr, weight_decay=1e-2)
        else:
            self.optimizer = torch.optim.AdamW([{'params': self.model.parameters()}], lr=config.lr,
                                               weight_decay=1e-2)


        self.start_epoch = 0
        self.best_epoch = 0
        self.best_metric = {}
        for task_id in self.config.task_id_list:
            if Indice_Column_NumCls_Dict[Task_List[task_id]]['metric'] == 'mse':
                self.best_metric[Task_List[task_id]] = 1000.0
            else:
                self.best_metric[Task_List[task_id]] = 0.01

        self.iter_counter = 0
        if checkpoint is not None:
            self.start_epoch = checkpoint['epoch']
            self.best_epoch = checkpoint['best_epoch']
            self.iter_counter = checkpoint['iter_counter']
            self.best_metric = checkpoint['best_metric']

            if self.config.backbone_update in ['ft',]:
                self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
                self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])

            elif self.config.backbone_update in ['fz','scratch','lora']:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.model.load_state_dict(checkpoint['state_dict'])

        if self.config.backbone_update in ['ft', ]:
            self.scheduler = ExponentialLR(optimizer=self.decoder_optimizer, gamma=config.lr_decay)

        elif self.config.backbone_update in ['fz', 'scratch', 'lora']:
            self.scheduler = ExponentialLR(optimizer=self.optimizer, gamma=config.lr_decay)  # CosineLRScheduler(self.optimizer, t_initial=self.config.total_epoch, cycle_limit=1, t_in_epochs=True) #ExponentialLR(optimizer=self.optimizer, gamma=config.lr_decay)

        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.test_loaders = test_loaders


        ## warm up optimizer
        self.num_tasks = len(config.task_id_list)
        # self.task_names = config.task_names

        self.config = config


        #self.epochs = config.epochs
        self.save_dir = save_dir
        self.logger = self._create_logger(save_dir)
        self.writer = tensorboard.SummaryWriter(os.path.join(save_dir, 'tensorboard_writer'))
        self.wrt_step = 0

        self.epsilon = 1e-8
        self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        self.imagenet_std = np.array([0.229, 0.224, 0.225])


    def train(self):
        #loss_seg_orig_dict = {}

        # warm up train seg
        #loss_seg_orig_warmseg = []

        self.logger.info(f'\n-------------------Start training: total {self.config.total_epoch} epoches; {self.config.Iter_Num} iters; {self.config.iter_per_epoch} per iter  ------------------------ ')
        self.logger.info(
            f'\n-------------------------dataset info: train {len(self.train_loaders.dataset)}; vali {len(self.val_loaders.dataset)}; test: {len(self.test_loaders.dataset)}  -------------------------- ')

        # scaler1 = GradScaler()
        loss_list = {}
        for task_id in self.config.task_id_list:
            task_name = Task_List[task_id]
            loss_list[task_name] = []

        loss_file = self.save_dir + '/loss_change.npy'
        if os.path.exists(loss_file):
            loss_list = np.load(loss_file, allow_pickle=True)

        rng = np.random.default_rng()
        grad_dims = []

        for param in self.model.get_base_params():
            grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), self.num_tasks).cuda()

        lambda_weight = np.ones([self.num_tasks, self.config.total_epoch])

        #init_lr = self.optimizer_all.param_groups[0]['lr']
        for epoch in range(self.start_epoch,self.config.total_epoch):#self.epochs

            dataloader = iter(self.train_loaders)
            train_starttime = datetime.datetime.now()


            self.model.train()

            for batch_idx in range(self.config.iter_per_epoch):
                main_task_lr = 0.0
                if self.config.backbone_update in ['ft', ]:
                    main_task_lr = self.decoder_optimizer.param_groups[0]['lr']

                elif self.config.backbone_update in ['fz', 'scratch', 'lora']:
                    main_task_lr = self.optimizer.param_groups[0]['lr']

                smaple_batch = next(dataloader)
                img = None

                if self.config.Train_Data == 'L':
                    img = smaple_batch['left_img']
                    # label = label_L
                elif self.config.Train_Data == 'R':
                    img = smaple_batch['right_img']

                img = img.cuda()
                for task_id in self.config.task_id_list:
                    if Indice_Column_NumCls_Dict[Task_List[task_id]]['num_cls'] == 1:
                        smaple_batch[Task_List[task_id]] = smaple_batch[Task_List[task_id]].float().cuda()
                    elif Indice_Column_NumCls_Dict[Task_List[task_id]]['num_cls'] == 2:
                        smaple_batch[Task_List[task_id]] = smaple_batch[Task_List[task_id]].float().cuda()
                    else:
                        smaple_batch[Task_List[task_id]] = smaple_batch[Task_List[task_id]].long().cuda()


                # with autocast():
                self.model.zero_grad()

                task_out = self.model(img)
                # print(task_out.size())

                train_loss = self.model_fit(task_out, smaple_batch)



                # task1_loss_list.append(task1_loss.item())
                # task2_loss_list.append(task2_loss.item())

                train_loss_tmp = [0 for i in range(self.num_tasks)]

                if self.config.weight == 'equal':
                    for i in range(self.num_tasks):
                        train_loss_tmp[i] = train_loss[i] * lambda_weight[i, epoch]

                if self.config.backbone_update in ['ft', ]:
                    self.encoder_optimizer.zero_grad()
                    self.decoder_optimizer.zero_grad()

                elif self.config.backbone_update in ['fz', 'scratch', 'lora']:
                    self.optimizer.zero_grad()

                if self.config.method == "graddrop":
                    for i in range(self.num_tasks):
                        if i < self.num_tasks-1:
                            train_loss_tmp[i].backward(retain_graph=True)
                        else:
                            train_loss_tmp[i].backward()
                        self.grad2vec(grads, grad_dims, i)
                        self.model.zero_grad_shared_modules()
                    g = self.graddrop(grads)
                    self.overwrite_grad(g, grad_dims)
                    if self.config.backbone_update in ['ft', ]:
                        self.encoder_optimizer.step()
                        self.decoder_optimizer.step()

                    elif self.config.backbone_update in ['fz', 'scratch', 'lora']:
                        self.optimizer.step()

                elif self.config.method == "pcgrad":
                    for i in range(self.num_tasks):
                        # aa = torch.isnan(train_loss_tmp[i])
                        # bb = torch.isinf(train_loss_tmp[i])
                        # self.logger.info(
                        #     f'\n task{i}: {aa},{bb}')
                        if i < self.num_tasks-1:
                            train_loss_tmp[i].backward(retain_graph=True)
                        else:
                            train_loss_tmp[i].backward()
                        self.grad2vec(grads, grad_dims, i)
                        self.model.zero_grad_shared_modules()
                    g = self.pcgrad(grads, rng)
                    self.overwrite_grad(g, grad_dims)
                    if self.config.backbone_update in ['ft', ]:
                        self.encoder_optimizer.step()
                        self.decoder_optimizer.step()

                    elif self.config.backbone_update in ['fz', 'scratch', 'lora']:
                        self.optimizer.step()
                elif self.config.method == "mgd":
                    for i in range(self.num_tasks):
                        if i < self.num_tasks-1:
                            train_loss_tmp[i].backward(retain_graph=True)
                        else:
                            train_loss_tmp[i].backward()
                        self.grad2vec(grads, grad_dims, i)
                        self.model.zero_grad_shared_modules()
                    g = self.mgd(grads)
                    self.overwrite_grad( g, grad_dims)
                    if self.config.backbone_update in ['ft', ]:
                        self.encoder_optimizer.step()
                        self.decoder_optimizer.step()

                    elif self.config.backbone_update in ['fz', 'scratch', 'lora']:
                        self.optimizer.step()
                elif self.config.method == "cagrad":
                    for i in range(self.num_tasks):
                        if i < self.num_tasks-1:
                            train_loss_tmp[i].backward(retain_graph=True)
                        else:
                            train_loss_tmp[i].backward()
                        self.grad2vec(grads, grad_dims, i)
                        self.model.zero_grad_shared_modules()
                    g = self.cagrad(grads, self.config.alpha, rescale=1)
                    self.overwrite_grad(g, grad_dims)
                    if self.config.backbone_update in ['ft', ]:
                        self.encoder_optimizer.step()
                        self.decoder_optimizer.step()

                    elif self.config.backbone_update in ['fz', 'scratch', 'lora']:
                        self.optimizer.step()


                # self.optimizer_Seg.step()
                # self.lr_scheduler_task.step()

                self.iter_counter += 1

                if self.iter_counter % self.config.lr_decay_iters == 0:
                    self.scheduler.step()



                # log
                if batch_idx % 10 == 0:
                    self.logger.info(f'\n Epoch: {epoch}, iter: {batch_idx}, model lr: {main_task_lr},  task loss:')
                    info  = ''
                    for i,task_id in enumerate(self.config.task_id_list):
                        info += f'{Task_List[task_id]}:{train_loss[i]}----'
                    self.logger.info(info)

                for i,task_id in enumerate(self.config.task_id_list):
                    task_name = Task_List[task_id]
                    loss_list[task_name].append(train_loss[i].to("cpu").item())

                del img,smaple_batch,task_out,train_loss,train_loss_tmp
                torch.cuda.empty_cache()


            train_endtime = datetime.datetime.now()  # MedMNIST_TEST_Tasks
            self.logger.info(f'\n epoch: {epoch},  train time: {(train_endtime - train_starttime).seconds}')

            num_evl = 20
            if self.config.total_epoch < num_evl:
                num_evl = self.config.total_epoch

            if (epoch % (self.config.total_epoch // num_evl) == 0) or (epoch == self.config.total_epoch - 1):
                prediction_metric_dict = Evaluate_UKBB_Index(self.config, self.val_loaders, self.model, self.save_dir,'val', self.config.Train_Data)

                upgrade_ratio = 0.0
                for task_id in self.config.task_id_list:
                    task_name = Task_List[task_id]
                    if Indice_Column_NumCls_Dict[task_name]['metric'] == 'mse':
                        upgrade_ratio += (self.best_metric[task_name] - prediction_metric_dict[task_name]['mean_m']) / self.best_metric[task_name]

                    else:
                        upgrade_ratio += (prediction_metric_dict[task_name]['f1_macro'] - self.best_metric[task_name]) / self.best_metric[task_name]

                if upgrade_ratio > 0.0:
                    for task_id in self.config.task_id_list:
                        task_name = Task_List[task_id]
                        if Indice_Column_NumCls_Dict[task_name]['metric'] == 'mse':
                            self.best_metric[task_name] = prediction_metric_dict[task_name]['mean_m']
                        else:
                            self.best_metric[task_name] =prediction_metric_dict[task_name]['f1_macro']

                    self.best_epoch = epoch + 1
                    best_model_file = os.path.join(self.save_dir, 'model_param.pkl')
                    if os.path.exists(best_model_file):
                        os.remove(os.path.join(self.save_dir, 'model_param.pkl'))
                    torch.save(self.model.state_dict(), best_model_file)

            checkpoint_save_file = os.path.join(self.save_dir, 'checkpoint.pth.tar')
            if os.path.exists(checkpoint_save_file):
                os.remove(checkpoint_save_file)

            if self.config.backbone_update in ['ft', ]:
                torch.save({'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'best_epoch': self.best_epoch,
                            'best_metric': self.best_metric,
                            'iter_counter': self.iter_counter,
                            'encoder_optimizer': self.encoder_optimizer.state_dict(),
                            'decoder_optimizer': self.decoder_optimizer.state_dict(),
                            # 'scaler_state_dict': self.scaler.state_dict()
                            },
                           self.save_dir + 'checkpoint.pth.tar')  # 'scaler_state_dict': scaler.state_dict(),  # Save scaler state



            elif self.config.backbone_update in ['fz', 'scratch', 'lora']:
                torch.save({'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'best_epoch': self.best_epoch,
                            'best_metric': self.best_metric,
                            'iter_counter': self.iter_counter,
                            'optimizer': self.optimizer.state_dict(),
                            # 'scaler_state_dict': self.scaler.state_dict()
                            }, self.save_dir + 'checkpoint.pth.tar')



        self.logger.info(f'\n-------------------------End training -------------------------- ')
        # save loss
        # np.save(self.save_dir + '/train_task1_losses.npy', np.array(task1_loss_list))
        # np.save(self.save_dir + '/train_task2_losses.npy', np.array(task2_loss_list))

        # save model
        self.logger.info(f'\n-------------------------Start Testing -------------------------- ')
        # test model, save results

        self.model.load_state_dict(torch.load(os.path.join(self.save_dir, 'model_param.pkl'), weights_only=True))
        acc_metric = Evaluate_UKBB_Index(self.config, self.test_loaders, self.model, self.save_dir, 'test', self.config.Train_Data)


        # save loss curve
        if len(loss_list[Task_List[0]]) != 0:
            # colors = cycler(color=['r', 'g', 'b', 'y', 'c'])  # Define custom colors
            #
            # # Set the color cycle to the plot
            # plt.gca().set_prop_cycle(colors)

            for task_id in self.config.task_id_list:
                task_name = Task_List[task_id]
                if len(loss_list[task_name]) > 10000:
                    loss_list[task_name] = loss_list[task_name][:10000]

                loss_np = np.array(loss_list[task_name])
                cumsum_main = np.cumsum(loss_np)
                main_avg = cumsum_main / (np.arange(1, len(loss_list) + 1))

                plt.plot(main_avg, label=r'$\mathcal{L}_{'+task_name+'}$')

            plt.title("Training Loss")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.legend()  # To show the legend
            plt.savefig(self.save_dir + "/train_loss.png")
            plt.close()

        self.logger.info(f'\n-------------------------End Testing -------------------------- ')


        for handler in self.logger.handlers:
            handler.close()  # Close each handler to release resources
            self.logger.removeHandler(handler)  # Remove the handler from the logger


    def _create_logger(self, save_dir):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logfile = os.path.join(save_dir, 'training.log')

        fh = logging.FileHandler(logfile, mode='a')
        fh.setLevel(logging.DEBUG)

        logger.addHandler(fh)

        return logger


    def graddrop(self, grads):
        P = 0.5 * (1. + grads.sum(1) / (grads.abs().sum(1)+1e-8))
        U = torch.rand_like(grads[:,0])
        M = P.gt(U).view(-1,1)*grads.gt(0) + P.lt(U).view(-1,1)*grads.lt(0)
        g = (grads * M.float()).mean(1)
        return g

    def mgd(self, grads):
        grads_cpu = grads.t().cpu()
        sol, min_norm = MinNormSolver.find_min_norm_element([
            grads_cpu[t] for t in range(grads.shape[-1])])
        w = torch.FloatTensor(sol).to(grads.device)
        g = grads.mm(w.view(-1, 1)).view(-1)
        return g

    def pcgrad(self, grads, rng):
        grad_vec = grads.t()
        num_tasks = self.num_tasks

        shuffled_task_indices = np.zeros((num_tasks, num_tasks - 1), dtype=int)
        for i in range(num_tasks):
            task_indices = np.arange(num_tasks)
            task_indices[i] = task_indices[-1]
            shuffled_task_indices[i] = task_indices[:-1]
            rng.shuffle(shuffled_task_indices[i])
        shuffled_task_indices = shuffled_task_indices.T

        normalized_grad_vec = grad_vec / (
            grad_vec.norm(dim=1, keepdim=True) + 1e-8
        )  # num_tasks x dim
        modified_grad_vec = deepcopy(grad_vec)
        for task_indices in shuffled_task_indices:
            normalized_shuffled_grad = normalized_grad_vec[
                task_indices
            ]  # num_tasks x dim
            dot = (modified_grad_vec * normalized_shuffled_grad).sum(
                dim=1, keepdim=True
            )  # num_tasks x dim
            modified_grad_vec -= torch.clamp_max(dot, 0) * normalized_shuffled_grad
        g = modified_grad_vec.mean(dim=0)
        return g

    def cagrad(self, grads, alpha=0.5, rescale=0):
        GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
        g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

        x_start = np.ones(self.num_tasks) / self.num_tasks
        bnds = tuple((0, 1) for x in x_start)
        cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
        A = GG.numpy()
        b = x_start.copy()
        c = (alpha * g0_norm + 1e-8).item()

        def objfn(x):
            return (x.reshape(1, self.num_tasks).dot(A).dot(b.reshape(self.num_tasks, 1)) + c * np.sqrt(
                x.reshape(1, self.num_tasks).dot(A).dot(x.reshape(self.num_tasks, 1)) + 1e-8)).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        gw = (grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = grads.mean(1) + lmbda * gw
        if rescale == 0:
            return g
        elif rescale == 1:
            return g / (1 + alpha ** 2)
        else:
            return g / (1 + alpha)

    def grad2vec(self, grads, grad_dims, task):
        # store the gradients
        grads[:, task].fill_(0.0)
        cnt = 0

        for p in self.model.get_base_params():
            grad = p.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, newgrad, grad_dims):
        newgrad = newgrad * self.num_tasks # to match the sum loss
        cnt = 0

        for param in self.model.get_base_params():
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

    def model_fit(self, x_pred, smaple_batch):
        loss_dict = []

        for i,task_id in enumerate(self.config.task_id_list):
            task_name = Task_List[task_id]
            x_output = smaple_batch[task_name]

            if Indice_Column_NumCls_Dict[Task_List[task_id]]['num_cls'] == 1:
                loss = F.smooth_l1_loss(x_pred[i].squeeze(1), x_output.squeeze())
                loss_dict.append(loss)
            elif Indice_Column_NumCls_Dict[Task_List[task_id]]['num_cls'] == 2:
                loss = F.binary_cross_entropy_with_logits(x_pred[i].squeeze(1), x_output.squeeze())
                loss_dict.append(loss)
            else:
                loss = F.cross_entropy(x_pred[i], x_output)
                loss_dict.append(loss)

        return loss_dict






