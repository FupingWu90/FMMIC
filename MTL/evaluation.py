# -*- coding:utf-8 -*-

import torch
from torch import nn
import numpy as np
from scipy import stats
from dataset import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score,confusion_matrix,r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def confusion_matrix_c(rater_a, rater_b, min_rating=None, max_rating=None):
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def drawing_cm_map(cm,xticklabels,yticklabels,task_name,save_path):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap='viridis',xticklabels=xticklabels, yticklabels=yticklabels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(os.path.join(save_path, '{}.png').format(task_name))
    plt.close()


def drawing_pred_map(pred,target,task_name,save_path):
    # Fit the model
    model = LinearRegression()
    model.fit(pred.reshape(-1,1), target)

    # Make predictions
    y_pred = model.predict(pred.reshape(-1,1))

    # Scatter plot
    plt.scatter(target, pred, color='blue', label='samples')

    # Regression line
    plt.plot(y_pred, pred, color='red', label='Regression line')

    # Identity line (dotted)
    plt.plot([target.min()-1, target.max()+1], [target.min()-1, target.max()+1], color='green', linestyle='--', label='Identity line')

    # Enhancements
    plt.title('Scatter plot between Predicted and Target Values')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.legend()

    # Calculate R^2
    r2 = r2_score(target, y_pred)
    plt.text(target.min(), pred.max(), f'R²: {r2:.2f}', fontsize=12)  # x, y defines the position of the text

    plt.savefig(os.path.join(save_path, '{}.png').format(task_name))
    plt.close()


def Evaluate_UKBB_Index(config,Test_Loaders,model,save_dir,mode,Train_data):
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    #img_size = config.Img_Size
    model.eval()
    if mode == 'test':
        params_num = count_parameters(model)
    else:
        params_num = count_parameters(model)


    soft_m = nn.Sigmoid()

    if mode != 'test':
        outcome_dict = {}
        prediction_metric_dict = {}
        for task_id in config.task_id_list:
            task_name = Task_List[task_id]
            if Indice_Column_NumCls_Dict[task_name]['metric'] == 'roc':
                outcome_dict[task_name] = {'bi_correct':0,'bi_prob_list':[],'bi_true_list':[],'bi_pred_left':[]}
                prediction_metric_dict[task_name] = {'acc':0.0,'roc':0.0,'f1_macro':0.0}

            elif Indice_Column_NumCls_Dict[task_name]['metric'] == 'macro_f1':
                outcome_dict[task_name] = {'pred_all': [], 'label_all': []}
                prediction_metric_dict[task_name] = {'acc_': 0.0, 'precision_macro': 0.0, 'precision_micro': 0.0,
                                                     'recall_macro': 0.0, 'recall_micro': 0.0, 'f1_macro': 0.0,
                                                     'f1_micro': 0.0, 'kappa_': 0.0, 'cm_': None,'ratios_class':None}

            else:
                outcome_dict[task_name] = {'error_list': []}
                prediction_metric_dict[task_name] = {'mean_m': 0.0, 'std_m': 0.0,}



        with torch.no_grad():  # operations inside don't track history
            #test_dataset = Test_Loaders

            for i, smaple_batch in enumerate(Test_Loaders):

                if Train_data == 'L':
                    img = smaple_batch['left_img']
                elif Train_data == 'R':
                    img = smaple_batch['right_img']

                img =img.cuda()

                test_pred_i = model(img)

                for i,task_id in enumerate(config.task_id_list):
                    task_name = Task_List[task_id]
                    if Indice_Column_NumCls_Dict[task_name]['metric'] == 'roc':

                        pred_prob_i = soft_m(test_pred_i[i]).squeeze().cpu()
                        y_pred_i = (pred_prob_i > 0.5)*1

                        outcome_dict[task_name]['bi_pred_left'] += y_pred_i.numpy().tolist()

                        outcome_dict[task_name]['bi_prob_list'] += pred_prob_i.numpy().tolist()
                        outcome_dict[task_name]['bi_true_list'] += smaple_batch[task_name].numpy().tolist()

                        outcome_dict[task_name]['bi_correct'] += y_pred_i.eq(smaple_batch[task_name].data).sum().item()

                    elif Indice_Column_NumCls_Dict[task_name]['metric'] == 'macro_f1':

                        _, truearg_ = torch.max(test_pred_i[i], 1, keepdim=False)
                        pred_i_ = np.squeeze(truearg_.detach().cpu().numpy())
                        outcome_dict[task_name]['pred_all'] += pred_i_.tolist()

                        outcome_dict[task_name]['label_all'] += smaple_batch[task_name].numpy().tolist()

                    else:
                        pred_prob_i = test_pred_i[i].squeeze().cpu()
                        MAE_i = (pred_prob_i - smaple_batch[task_name]).abs().numpy().tolist()

                        outcome_dict[task_name]['error_list'] = outcome_dict[task_name]['error_list'] + MAE_i




        if Train_data == 'LR_Mix':
            total_num = len(Test_Loaders.dataset) *2
        else:
            total_num = len(Test_Loaders.dataset)

        for task_id in config.task_id_list:
            task_name = Task_List[task_id]
            if Indice_Column_NumCls_Dict[task_name]['metric'] == 'roc':
                prediction_metric_dict[task_name]['acc'] = 100. * outcome_dict[task_name]['bi_correct'] / total_num
                prediction_metric_dict[task_name]['roc'] = roc_auc_score(np.array(outcome_dict[task_name]['bi_true_list']), np.array(outcome_dict[task_name]['bi_prob_list']))
                prediction_metric_dict[task_name]['f1_macro'] = f1_score(np.array(outcome_dict[task_name]['bi_true_list']), np.array(outcome_dict[task_name]['bi_pred_left']), average='macro')

            elif Indice_Column_NumCls_Dict[task_name]['metric'] == 'macro_f1':
                prediction_metric_dict[task_name]['acc_'] = accuracy_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['pred_all']))
                prediction_metric_dict[task_name]['precision_macro'] = precision_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['pred_all']), average='macro')
                prediction_metric_dict[task_name]['precision_micro'] = precision_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['pred_all']), average='micro')
                prediction_metric_dict[task_name]['recall_macro'] = recall_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['pred_all']), average='macro')
                prediction_metric_dict[task_name]['recall_micro'] = recall_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['pred_all']), average='micro')
                prediction_metric_dict[task_name]['f1_macro'] = f1_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['pred_all']), average='macro')
                prediction_metric_dict[task_name]['f1_micro'] = f1_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['pred_all']), average='micro')
                prediction_metric_dict[task_name]['kappa_'] = cohen_kappa_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['pred_all']))
                prediction_metric_dict[task_name]['cm_'] = confusion_matrix(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['pred_all']),
                                       labels=[i for i in range(len(Indice_Column_NumCls_Dict[task_name]['label_names']))])

                # statistics
                num_samples_class = []
                for i in range(len(Indice_Column_NumCls_Dict[task_name]['label_names'])):
                    index_sample_class = (np.array(outcome_dict[task_name]['label_all']) == i) * 1
                    num_samples_class.append(index_sample_class.sum())

                prediction_metric_dict[task_name]['ratios_class'] = [float(i) / len(outcome_dict[task_name]['label_all']) for i in num_samples_class]

            else:
                prediction_metric_dict[task_name]['mean_m'] = np.mean(outcome_dict[task_name]['error_list'])
                prediction_metric_dict[task_name]['std_m'] = np.std(outcome_dict[task_name]['error_list'])


        with open("%s/testout_index.txt" % (save_dir), "a") as f:
            f.writelines(
                ["\n\n\n", mode, ":","\n","params: ", str(params_num), "test", ":", ",total number: ", str(total_num),
                 ])

            for task_id in config.task_id_list:
                task_name = Task_List[task_id]
                f.writelines(["\n","---------(",str(task_id), ")---------", task_name,"------metrics------","\n","------"])
                for key,values in  prediction_metric_dict[task_name].items():
                    f.writelines([ key,":",str(values),"------"])


        return prediction_metric_dict

    else: # mode test
        if Train_data != 'LR_Combine':
            outcome_dict = {}
            prediction_metric_dict = {}
            for task_id in config.task_id_list:
                task_name = Task_List[task_id]
                if Indice_Column_NumCls_Dict[task_name]['metric'] == 'roc':
                    outcome_dict[task_name] = {'bi_correct_left': [], 'bi_correct_right': [], 'bi_pred_left': [],
                                               'bi_pred_right': [], 'bi_prob_list_R': [],
                                               'bi_prob_list_L': [],'label_all': []}
                    prediction_metric_dict[task_name] = {'acc_L': None, 'roc_L': None, 'acc_L1': None,'precision_L_macro': None, 'precision_L_micro': None, 'recall_L_macro': None,
                                                         'recall_L_micro': None, 'f1_L_macro': None, 'f1_L_micro': None,'kappa_L': None, 'cm_L': None,
                                                         'acc_R': None, 'roc_R': None, 'acc_R1': None,'precision_R_macro': None, 'precision_R_micro': None, 'recall_R_macro': None,
                                                         'recall_R_micro': None, 'f1_R_macro': None, 'f1_R_micro': None,'kappa_R': None, 'cm_R': None,
                                                         'acc_all':None,'roc_all':None,'ratio_1':None,'ratio_0':None,'roc_avg':None,}

                elif Indice_Column_NumCls_Dict[task_name]['metric'] == 'macro_f1':
                    outcome_dict[task_name] = {'pred_left': [], 'pred_right': [], 'label_all': []}
                    prediction_metric_dict[task_name] = {'acc_L': None,
                                                         'precision_L_macro': None, 'precision_L_micro': None,
                                                         'recall_L_macro': None,
                                                         'recall_L_micro': None, 'f1_L_macro': None, 'f1_L_micro': None,
                                                         'kappa_L': None, 'cm_L': None,
                                                         'acc_R': None,
                                                         'precision_R_macro': None, 'precision_R_micro': None,
                                                         'recall_R_macro': None,
                                                         'recall_R_micro': None, 'f1_R_macro': None, 'f1_R_micro': None,
                                                         'kappa_R': None, 'cm_R': None,
                                                         'ratios_class': None, }

                else:
                    outcome_dict[task_name] = {'pred_list_left': [], 'pred_list_right': [], 'label_all': []}
                    prediction_metric_dict[task_name] = {}





            with torch.no_grad():  # operations inside don't track history
                # test_dataset = Test_Loaders

                for i, smaple_batch in enumerate(Test_Loaders):

                    # left eye
                    img_L = smaple_batch['left_img'].cuda()

                    test_pred_i_L = model(img_L)

                    # right eye
                    img_R = smaple_batch['right_img'].cuda()

                    test_pred_i_R = model(img_R)

                    for i,task_id in enumerate(config.task_id_list):
                        task_name = Task_List[task_id]
                        if Indice_Column_NumCls_Dict[task_name]['metric'] == 'roc':
                            pred_prob_i_L = soft_m(test_pred_i_L[i]).squeeze().cpu()
                            y_pred_i_L = (pred_prob_i_L > 0.5) * 1
                            outcome_dict[task_name]['bi_pred_left'] += y_pred_i_L.numpy().tolist()

                            outcome_dict[task_name]['bi_prob_list_L'] += pred_prob_i_L.numpy().tolist()

                            #bi_correct_left += y_pred_i_L.eq(label_L.data).sum().item()
                            true_index_L = (y_pred_i_L == smaple_batch[task_name].data)*1
                            outcome_dict[task_name]['bi_correct_left'] += true_index_L.numpy().tolist()
                            outcome_dict[task_name]['label_all'] += smaple_batch[task_name].numpy().tolist()


                            pred_prob_i_R = soft_m(test_pred_i_R[i]).squeeze().cpu()
                            y_pred_i_R = (pred_prob_i_R > 0.5) * 1
                            outcome_dict[task_name]['bi_pred_right'] += y_pred_i_R.numpy().tolist()

                            outcome_dict[task_name]['bi_prob_list_R'] += pred_prob_i_R.numpy().tolist()

                            #bi_correct_right += y_pred_i_R.eq(label_R.data).sum().item()
                            true_index_R = (y_pred_i_R == smaple_batch[task_name].data) * 1
                            outcome_dict[task_name]['bi_correct_right'] += true_index_R.numpy().tolist()


                        elif Indice_Column_NumCls_Dict[task_name]['metric'] == 'macro_f1':

                            _, truearg_L = torch.max(test_pred_i_L[i], 1, keepdim=False)
                            pred_i_L = np.squeeze(truearg_L.detach().cpu().numpy())
                            outcome_dict[task_name]['pred_left'] += pred_i_L.tolist()

                            outcome_dict[task_name]['label_all'] += smaple_batch[task_name].numpy().tolist()

                            # right eye

                            _, truearg_R = torch.max(test_pred_i_R[i], 1, keepdim=False)
                            pred_i_R = np.squeeze(truearg_R.detach().cpu().numpy())
                            outcome_dict[task_name]['pred_right'] += pred_i_R.tolist()

                        else:

                            pred_prob_i_L = test_pred_i_L[i].squeeze().cpu()
                            # MAE_i_L = (pred_prob_i_L - label_L).abs().numpy().tolist()
                            outcome_dict[task_name]['pred_list_left'] = outcome_dict[task_name]['pred_list_left'] + pred_prob_i_L.numpy().tolist()
                            outcome_dict[task_name]['label_all'] += smaple_batch[task_name].numpy().tolist()

                            pred_prob_i_R = test_pred_i_R[i].squeeze().cpu()
                            # MAE_i_L = (pred_prob_i_L - label_L).abs().numpy().tolist()
                            outcome_dict[task_name]['pred_list_right'] = outcome_dict[task_name]['pred_list_right'] + pred_prob_i_R.numpy().tolist()


            # save
            torch.save(outcome_dict, save_dir + 'outcome_dict.npy')
            torch.save(prediction_metric_dict, save_dir + 'prediction_metric_dict.npy')

            for task_id in config.task_id_list:
                task_name = Task_List[task_id]
                if Indice_Column_NumCls_Dict[task_name]['metric'] == 'roc':

                    prediction_metric_dict[task_name]['acc_L'] = 100. * sum(outcome_dict[task_name]['bi_correct_left'])/len(outcome_dict[task_name]['bi_correct_left'])
                    prediction_metric_dict[task_name]['roc_L'] = roc_auc_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['bi_prob_list_L']))

                    prediction_metric_dict[task_name]['acc_L1'] = accuracy_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['bi_pred_left']))
                    prediction_metric_dict[task_name]['precision_L_macro'] = precision_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['bi_pred_left']), average='macro')
                    prediction_metric_dict[task_name]['precision_L_micro'] = precision_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['bi_pred_left']), average='micro')
                    prediction_metric_dict[task_name]['recall_L_macro'] = recall_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['bi_pred_left']), average='macro')
                    prediction_metric_dict[task_name]['recall_L_micro'] = recall_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['bi_pred_left']), average='micro')
                    prediction_metric_dict[task_name]['f1_L_macro'] = f1_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['bi_pred_left']), average='macro')
                    prediction_metric_dict[task_name]['f1_L_micro'] = f1_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['bi_pred_left']), average='micro')
                    prediction_metric_dict[task_name]['kappa_L'] = cohen_kappa_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['bi_pred_left']))
                    cm_L = confusion_matrix(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['bi_pred_left']),labels=[0,1])
                    prediction_metric_dict[task_name]['cm_L'] = cm_L.tolist()
                    drawing_cm_map(cm_L, Indice_Column_NumCls_Dict[task_name]['label_names'], Indice_Column_NumCls_Dict[task_name]['label_names'], task_name+'_cm_map_left', save_dir)

                    # acc right eye
                    prediction_metric_dict[task_name]['acc_R'] = 100. * sum(outcome_dict[task_name]['bi_correct_right']) / len(outcome_dict[task_name]['bi_correct_right'])
                    prediction_metric_dict[task_name]['roc_R'] = roc_auc_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['bi_prob_list_R']))

                    prediction_metric_dict[task_name]['acc_R1'] = accuracy_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['bi_pred_right']))
                    prediction_metric_dict[task_name]['precision_R_macro'] = precision_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['bi_pred_right']), average='macro')
                    prediction_metric_dict[task_name]['precision_R_micro'] = precision_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['bi_pred_right']), average='micro')
                    prediction_metric_dict[task_name]['recall_R_macro'] = recall_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['bi_pred_right']), average='macro')
                    prediction_metric_dict[task_name]['recall_R_micro'] = recall_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['bi_pred_right']), average='micro')
                    prediction_metric_dict[task_name]['f1_R_macro'] = f1_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['bi_pred_right']), average='macro')
                    prediction_metric_dict[task_name]['f1_R_micro'] = f1_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['bi_pred_right']), average='micro')
                    prediction_metric_dict[task_name]['kappa_R'] = cohen_kappa_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['bi_pred_right']))
                    cm_R = confusion_matrix(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['bi_pred_right']), labels=[0, 1])
                    prediction_metric_dict[task_name]['cm_R'] = cm_R.tolist()
                    drawing_cm_map(cm_R, Indice_Column_NumCls_Dict[task_name]['label_names'], Indice_Column_NumCls_Dict[task_name]['label_names'], task_name+'_cm_map_right', save_dir)


                    # acc all
                    prediction_metric_dict[task_name]['acc_all'] = (prediction_metric_dict[task_name]['acc_L']+prediction_metric_dict[task_name]['acc_R'])/2
                    prediction_metric_dict[task_name]['roc_all'] = roc_auc_score(np.array(outcome_dict[task_name]['label_all'] + outcome_dict[task_name]['label_all']),
                                            np.array(outcome_dict[task_name]['bi_prob_list_R'] + outcome_dict[task_name]['bi_prob_list_L']))
                    # statistics
                    prediction_metric_dict[task_name]['ratio_1'] = 100. * sum(outcome_dict[task_name]['label_all'])/len(outcome_dict[task_name]['label_all'])
                    prediction_metric_dict[task_name]['ratio_0'] = 100.0-prediction_metric_dict[task_name]['ratio_1']
                    # statistic test
                    prediction_metric_dict[task_name]['roc_avg'] = roc_auc_score(np.array(outcome_dict[task_name]['label_all']),
                                            0.5 * (np.array(outcome_dict[task_name]['bi_prob_list_R']) + np.array(outcome_dict[task_name]['bi_prob_list_L'])))

                elif Indice_Column_NumCls_Dict[task_name]['metric'] == 'macro_f1':
                    prediction_metric_dict[task_name]['acc_L'] = accuracy_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['pred_left']))
                    prediction_metric_dict[task_name]['precision_L_macro'] = precision_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['pred_left']), average='macro')
                    prediction_metric_dict[task_name]['precision_L_micro'] = precision_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['pred_left']), average='micro')
                    prediction_metric_dict[task_name]['recall_L_macro'] = recall_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['pred_left']), average='macro')
                    prediction_metric_dict[task_name]['recall_L_micro'] = recall_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['pred_left']), average='micro')
                    prediction_metric_dict[task_name]['f1_L_macro'] = f1_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['pred_left']), average='macro')
                    prediction_metric_dict[task_name]['f1_L_micro'] = f1_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['pred_left']), average='micro')
                    prediction_metric_dict[task_name]['kappa_L'] = cohen_kappa_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['pred_left']))
                    cm_L = confusion_matrix(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['pred_left']),
                                            labels=[i for i in range(len(Indice_Column_NumCls_Dict[task_name]['label_names']))])
                    prediction_metric_dict[task_name]['cm_L'] = cm_L.tolist()
                    drawing_cm_map(cm_L, Indice_Column_NumCls_Dict[task_name]['label_names'], Indice_Column_NumCls_Dict[task_name]['label_names'], task_name+'_cm_map_left', save_dir)

                    # acc right eye

                    prediction_metric_dict[task_name]['acc_R'] = accuracy_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['pred_right']))
                    prediction_metric_dict[task_name]['precision_R_macro'] = precision_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['pred_right']), average='macro')
                    prediction_metric_dict[task_name]['precision_R_micro'] = precision_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['pred_right']), average='micro')
                    prediction_metric_dict[task_name]['recall_R_macro'] = recall_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['pred_right']), average='macro')
                    prediction_metric_dict[task_name]['recall_R_micro'] = recall_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['pred_right']), average='micro')
                    prediction_metric_dict[task_name]['f1_R_macro'] = f1_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['pred_right']), average='macro')
                    prediction_metric_dict[task_name]['f1_R_micro'] = f1_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['pred_right']), average='micro')
                    prediction_metric_dict[task_name]['kappa_R'] = cohen_kappa_score(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['pred_right']))
                    cm_R = confusion_matrix(np.array(outcome_dict[task_name]['label_all']), np.array(outcome_dict[task_name]['pred_right']),
                                            labels=[i for i in range(len(Indice_Column_NumCls_Dict[task_name]['label_names']))])
                    prediction_metric_dict[task_name]['cm_R'] = cm_R.tolist()
                    drawing_cm_map(cm_R, Indice_Column_NumCls_Dict[task_name]['label_names'], Indice_Column_NumCls_Dict[task_name]['label_names'], task_name+'_cm_map_right', save_dir)

                    # statistics
                    num_samples_class = []
                    for i in range(len(Indice_Column_NumCls_Dict[task_name]['label_names'])):
                        index_sample_class = (np.array(outcome_dict[task_name]['label_all']) == i) * 1
                        num_samples_class.append(index_sample_class.sum())

                    prediction_metric_dict[task_name]['ratios_class'] = [float(i) / len(outcome_dict[task_name]['label_all']) for i in num_samples_class]

                else:
                    left_error_numpy = np.abs((np.array(outcome_dict[task_name]['pred_list_left']) - np.array(outcome_dict[task_name]['label_all'])))
                    prediction_metric_dict[task_name]['mean_left'] = np.mean(left_error_numpy)
                    prediction_metric_dict[task_name]['std_left'] = np.std(left_error_numpy)
                    drawing_pred_map(np.array(outcome_dict[task_name]['pred_list_left']), np.array(outcome_dict[task_name]['label_all']), task_name+'_predict_left_true_value_plot',
                                     save_dir)

                    # right error
                    right_error_numpy = np.abs((np.array(outcome_dict[task_name]['pred_list_right']) - np.array(outcome_dict[task_name]['label_all'])))
                    prediction_metric_dict[task_name]['mean_right'] = np.mean(right_error_numpy)
                    prediction_metric_dict[task_name]['std_right'] = np.std(right_error_numpy)
                    drawing_pred_map(np.array(outcome_dict[task_name]['pred_list_right']), np.array(outcome_dict[task_name]['label_all']), task_name+'_predict_right_true_value_plot',
                                     save_dir)

                    # overall error
                    overall_error_numpy = np.concatenate((left_error_numpy, right_error_numpy), axis=0)
                    prediction_metric_dict[task_name]['mean_overall'] = np.mean(overall_error_numpy)
                    prediction_metric_dict[task_name]['std_overall'] = np.std(overall_error_numpy)

                    # average error
                    average_error_numpy = np.abs(
                        ((np.array(outcome_dict[task_name]['pred_list_left']) + np.array(outcome_dict[task_name]['pred_list_right'])) / 2 - np.array(outcome_dict[task_name]['label_all'])))
                    prediction_metric_dict[task_name]['mean_average'] = np.mean(average_error_numpy)
                    prediction_metric_dict[task_name]['std_average'] = np.std(average_error_numpy)
                    # statistic
                    prediction_metric_dict[task_name]['data_mean'] = np.mean(np.array(outcome_dict[task_name]['label_all']))
                    prediction_metric_dict[task_name]['data_std'] = np.std(np.array(outcome_dict[task_name]['label_all']))
                    # statistic test
                    prediction_metric_dict[task_name]['p_value'] = stats.ttest_rel(np.array(outcome_dict[task_name]['pred_list_left']), np.array(outcome_dict[task_name]['pred_list_right'])).pvalue

            with open("%s/testout_index.txt" % (save_dir), "a") as f:
                f.writelines(
                    ["\n\n\n", mode, ":", "\n", "params: ", str(params_num), ":",])

                for task_id in config.task_id_list:
                    task_name = Task_List[task_id]
                    f.writelines(["\n", "---------(",str(task_id), ")---------", task_name, "------metrics------", "\n", "------"])
                    for key, values in prediction_metric_dict[task_name].items():
                        f.writelines([key, ":", str(values),"------"])

            return prediction_metric_dict





