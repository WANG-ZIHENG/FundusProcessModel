#encoding=utf-8
import torch
import torchvision
import time
from torch import nn
from torch import optim
from torch.utils import data

from utils import TwoCropTransform
from Validate import validate_net
from Test import test_net
from misc import print_metrics, training_curve 
from PIL import Image
import os
import re
import argparse
from collections import defaultdict
import numpy as np
import logging
import csv
from torchvision import transforms, datasets, models
import sklearn.metrics as mtc
import mydatasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tqdm import tqdm
from losses import LDAMLoss, LogitAdjust, SupConLoss, MixLoss
import torch.nn.functional as F
import wandb
from get_features import Centers
from utils import get_datasets
import copy
from sklearn.metrics import cohen_kappa_score,roc_auc_score
from evaluator import getAUC,getACC
###########################
# Checking if GPU is used
###########################




########################################
# Setting basic parameters for the model
########################################

def get_args():
    parser=argparse.ArgumentParser(description='Train the model on images and target labels',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e','--epochs', metavar='E', type=int, default=60, nargs='?', help='Number of epochs', dest='max_epochs')
    parser.add_argument('-b','--batch-size', metavar='B', type=int, default=24, nargs='?', help='Batch size', dest='batch_size')
    parser.add_argument('-l','--learning-rate', metavar='LR', type=float, default=0.0001, nargs='?', help='Learning rate', dest='lr')
    parser.add_argument('--dataset', type=str, default="GastroVision", help='GastroVision or Ulcerative or OCT2017 or chest_xray or XXXMNIST')
    #https://github.com/MedMNIST/MedMNIST/tree/main?tab=readme-ov-file
    #breastmnist
    parser.add_argument('--supcon_loss_use', type=bool, default=False)
    parser.add_argument('--Logit_loss_use', type=bool, default=False)
    parser.add_argument('--LDAM_loss_use', type=bool, default=False)
    parser.add_argument('--CE_loss_use',  default=True ,action='store_true')
    parser.add_argument('--supcon_loss_weight', type=float, default=0.5)
    parser.add_argument('--Logit_loss_weight', type=float, default=1)
    parser.add_argument('--LDAM_loss_weight', type=float, default=1)
    parser.add_argument('--CE_loss_weight', type=float, default=1)
    parser.add_argument('--CCL_loss_use',  default=True ,action='store_true')
    parser.add_argument('--CCL_loss_weight', type=float, default=2)
    parser.add_argument('--WCE_loss_use', type=bool, default=False)
    parser.add_argument('--WCE_loss_weight', type=float, default=1)
    parser.add_argument('--device',type=str, default='cuda')
    parser.add_argument('--CE_sampling',type=str, default='double',help="double or single",choices=("single",'double'))
    parser.add_argument('--CCL_sampling',type=str, default='single',help="double or single",choices=("single",'double'))
    parser.add_argument('--model',type=str, default='DenseNet121',choices=("DenseNet121",'resnet18', 'resnet34', 'resnet50', 'resnext50_32x4d'))
    parser.add_argument('--pretrain_model', default=True,action='store_true')
    parser.add_argument('--data_reduce_rate', type=float, default=1,
                        help='Rate to reduce the dataset size by class (e.g., 0.9 for 90% of original size)')

    return parser.parse_args()
         

args=get_args()
batch_size=args.batch_size
max_epochs=args.max_epochs
lr=args.lr
device = args.device


model_path=r'./checkpoints/'  # set path to the folder that will store model's checkpoints




global val_micro_f1_max
global val_macro_f1_max


try:
   if not os.path.exists(os.path.dirname(model_path)):
       os.makedirs(os.path.dirname(model_path))
except OSError as err:
   print(err)

print("Directory '% s' created" % model_path)
filename='results_e'+str(max_epochs)+'_'+'b'+str(batch_size)+'_'+'lr'+str(lr)+'_'+args.model   #filename used for saving epoch-wise training details and test results

####################################
# Training
####################################



class Model(nn.Module):
    def __init__(self,args,n_classes, pretrained=False):
        super().__init__()
        if args.model == "DenseNet121":
            model = torchvision.models.densenet121(weights=pretrained).to(device)
        elif args.model == "resnet18":
            model = torchvision.models.resnet18(weights=pretrained).to(device)
        elif args.model == "resnet34":
            model = torchvision.models.resnet34(weights=pretrained).to(device)
        elif args.model == "resnet50":
            model = torchvision.models.resnet50(weights=pretrained).to(device)
        elif args.model == "resnext50_32x4d":
            model = torchvision.models.resnext50_32x4d(weights=pretrained).to(device)

        self.features = nn.ModuleList(model.children())[:-1]
        self.features = nn.Sequential(*self.features)
        if args.model == "DenseNet121":
            n_inputs = model.classifier.in_features
        else:
            n_inputs = model.fc.in_features
        self.normal_classifier = nn.Sequential(
            nn.Linear(n_inputs, n_classes),
            )
        self.ce_classifier = nn.Sequential(
            nn.Linear(n_inputs, n_classes),
        )
        self.args = args
    def forward(self, input_imgs,cal_center=False):
        features = self.features(input_imgs)
        fea = F.relu(features, inplace=True)
        fea = F.adaptive_avg_pool2d(fea, (1, 1))
        fea = torch.flatten(fea, 1)
        normal_fea = F.normalize(fea, dim=1)
        ce_output = self.ce_classifier(fea)
        normal_output= self.normal_classifier(normal_fea)
        if cal_center:
            return features, normal_output, ce_output
        else:

            bsz = input_imgs.shape[0]//2
            f1, f2 = torch.split(normal_output, [bsz, bsz], dim=0)
            nor_out1, nor_out2 = torch.split(normal_output, [bsz, bsz], dim=0)
            ce_out1, ce_out2 = torch.split(ce_output, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            normal_output = (nor_out1 + nor_out1) / 2
            if self.args.CE_sampling == "single":
                ce_output = ce_out1
            else:
                ce_output = (ce_out1 + ce_out2) / 2
            if self.args.CCL_sampling == "single":
                fea_pair_mean = f1
            else:
                fea_pair_mean = (f1 + f2) / 2

            return features,normal_output,ce_output,fea_pair_mean

def add_prefix(dct, prefix):
    return {f'{prefix}-{key}': val for key, val in dct.items()}


class train:
    def __init__(self):
        self.args = get_args()
        training_dataset, test_dataset, validation_dataset = get_datasets(self.args)
        self.cls_num_list = training_dataset.cls_num_list
        self.n_classes = len(self.cls_num_list)
        self.training_generator=data.DataLoader(training_dataset,batch_size,shuffle=True,num_workers=8) # ** unpacks a dictionary into keyword arguments
        self.validation_generator=data.DataLoader(validation_dataset,batch_size,shuffle=False,num_workers=8)
        self.test_generator=data.DataLoader(test_dataset,batch_size,num_workers=8)
        if args.data_reduce_rate == 1.0:
            print("Number of Each Class of Training set images:{}".format(training_dataset.cls_num_list))
        print('Number of Training set images:{}'.format(len(training_dataset)))
        print('Number of Validation set images:{}'.format(len(validation_dataset)))
        print('Number of Test set images:{}'.format(len(test_dataset)))
        # Initialize model
        self.model = Model(args, n_classes=self.n_classes, pretrained=args.pretrain_model)  # make weights=True if you want to download pre-trained weights
        # model.load_state_dict(torch.load('./densenet121.pth',map_location='cuda'))   # provide a .pth path for already downloaded weights; otherwise comment this line out
        # Option to freeze model weights
        for param in self.model.parameters():
            param.requires_grad = True  # Set param.requires_grad = False if you want to train only the last updated layers and freeze all other layers


        self.center = Centers(training_dataset=copy.deepcopy(training_dataset),model=copy.deepcopy(self.model),args=self.args,device=device)
        self.model.to(device)
    def train_net(self):
        model = self.model


        optimizer=optim.Adam(model.parameters(), lr, weight_decay=1e-4)
        scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=4,verbose=True)

        criterion = MixLoss(cls_num_list=self.cls_num_list,args = self.args)







        val_micro_f1_max=0.0
        val_macro_f1_max=0.0
        epochs=[]
        lossesT=[]
        lossesV=[]

        for epoch in range(max_epochs):
            print('Epoch {}/{}'.format(epoch+1,max_epochs))
            print('-'*10)
            
            since=time.time()
            train_metrics=defaultdict(float)
            total_loss=0
            total_ldam_loss =0
            total_logit_loss=0
            total_ce_loss =0
            total_supcon_loss =0
            total_ccl_loss = 0
            running_corrects=0
            num_steps=0
            
            all_labels_d = torch.tensor([], dtype=torch.long).to(device)
            all_predictions_d = torch.tensor([], dtype=torch.long).to(device)
            all_predictions_probabilities_d = []
            all_softmax_output = []
            # loss_weight_alpha = 1 - (epoch/max_epochs)**2
            # args.CE_loss_weight, args.CCL_loss_weight = loss_weight_alpha, (1-loss_weight_alpha)

            model.train()
            
            #Training
            for image, labels in tqdm(self.training_generator):
                #Transfer to GPU:

                images = torch.cat([image[0], image[1]], dim=0)

                if torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                bsz = labels.shape[0]

                # compute loss
                features,normal_output,ce_output,fea_pair_mean = model(images)


                with torch.no_grad():
                    class_centers = self.center.update_class_centers(epoch, fea_pair_mean.detach(), labels)

                loss, ldam_loss, logit_loss, ce_loss, supcon_loss, ccl_loss,wce_loss= criterion(features, normal_output, ce_output,
                                                                              labels, fea_pair_mean, class_centers)

                #loss,ldam_loss,logit_loss,ce_loss,supcon_loss = criterion(features,normal_output,ce_output, labels)

                if (args.CE_loss_use== True or args.WCE_loss_use== True) and args.LDAM_loss_use == False and args.Logit_loss_use == False:
                    predicted_probability, predicted = torch.max(ce_output, dim=1)
                    softmax_output = F.softmax(ce_output,dim=1)
                elif (args.CE_loss_use== False or args.WCE_loss_use== False) and (args.LDAM_loss_use == True or args.Logit_loss_use == True) :

                    predicted_probability, predicted = torch.max(normal_output, dim=1)
                    softmax_output = F.softmax(normal_output,dim=1)
                else:
                    output = (ce_output + normal_output)/2
                    predicted_probability, predicted = torch.max(output, dim=1)
                    softmax_output = F.softmax(output,dim=1)

                num_steps+=bsz
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss+=loss.item()*bsz
                total_ldam_loss += ldam_loss.item()*bsz
                total_logit_loss += logit_loss.item()*bsz
                total_ce_loss += ce_loss.item()*bsz
                total_supcon_loss += supcon_loss.item()*bsz
                total_ccl_loss += ccl_loss.item()*bsz


                running_corrects += torch.sum(predicted == labels.data)
                all_labels_d = torch.cat((all_labels_d, labels), 0)
                all_predictions_d = torch.cat((all_predictions_d, predicted), 0)
                all_softmax_output.append(softmax_output.cpu().detach().numpy())
                
            y_true = all_labels_d.cpu()
            y_predicted = all_predictions_d.cpu()  # to('cpu')
            all_softmax_output = np.concatenate(all_softmax_output)
            
            #############################
            # Standard metrics 
            #############################
            # 计算QWK指标
            train_qwk_score = cohen_kappa_score(y_true, y_predicted, weights='quadratic')
            train_micro_precision=mtc.precision_score(y_true, y_predicted, average="micro")     
            train_micro_recall=mtc.recall_score(y_true, y_predicted, average="micro")
            train_micro_f1=mtc.f1_score(y_true, y_predicted, average="micro")  
        
            train_macro_precision=mtc.precision_score(y_true, y_predicted, average="macro")     
            train_macro_recall=mtc.recall_score(y_true, y_predicted, average="macro")
            train_macro_f1=mtc.f1_score(y_true, y_predicted, average="macro")  
        
            train_mcc=mtc.matthews_corrcoef(y_true, y_predicted)

            y_true = y_true.detach().numpy()
            acc = getACC(y_true, all_softmax_output, task=self.training_generator.dataset.task)
            if self.training_generator.dataset.task == 'binary-class':
                all_softmax_output = np.max(all_softmax_output, axis=1)
            auc = getAUC(y_true, all_softmax_output, task=self.training_generator.dataset.task)
            train_metrics['acc'] =acc
            train_metrics['auc'] =auc
            train_metrics['loss']=total_loss/num_steps
            train_metrics['ldam_loss']=total_ldam_loss/num_steps
            train_metrics['logit_loss']=total_logit_loss/num_steps
            train_metrics['ce_loss']=total_ce_loss/num_steps
            train_metrics['supcon_loss']=total_supcon_loss/num_steps
            train_metrics['ccl_loss'] = total_ccl_loss / num_steps



        
            train_metrics['micro_precision']=train_micro_precision
            train_metrics['micro_recall']=train_micro_recall
            train_metrics['micro_f1']=train_micro_f1
            train_metrics['macro_precision']=train_macro_precision
            train_metrics['macro_recall']=train_macro_recall
            train_metrics['macro_f1']=train_macro_f1
            train_metrics['mcc']=train_mcc
            train_metrics['qwk'] = train_qwk_score
            
            print('Training...')
            print('Train_loss:{:.3f}'.format(total_loss/num_steps))
           
            
            print_metrics(train_metrics,num_steps)
            wandb.log(add_prefix(train_metrics, f'train'), step=epoch, commit=False)

            ############################
            # Validation
            ############################
            
            model.eval()
            with torch.no_grad():
                val_loss, val_metrics, val_num_steps=validate_net(epoch,model,self.validation_generator,device,criterion,args,self.center)
                #val_loss, val_metrics, val_num_steps=validate_net(model,self.validation_generator,device,criterion,args)
                
            scheduler.step(val_loss)
            epochs.append(epoch)
            lossesT.append(total_loss/num_steps)
            lossesV.append(val_loss)
            
            print('.'*5)
            print('Validating...')
            print('val_loss:{:.3f}'.format(val_loss))
        
            print_metrics(val_metrics,val_num_steps)


            ##################################################################
            # Writing epoch-wise training and validation results to a csv file 
            ##################################################################

            key_name=['Epoch','Train_loss','Train_micro_precision','Train_micro_recall','Train_micro_f1','Train_macro_precision','Train_macro_recall','Train_macro_f1','Train_mcc','Val_loss','Val_micro_precision','Val_micro_recall','Val_micro_f1','Val_macro_precision','Val_macro_recall','Val_macro_f1','Val_mcc']
            train_list=[]
            train_list.append(epoch)

            try:

                with open(filename+str('.csv'), 'a',newline="") as f:
                    wr = csv.writer(f,delimiter=",")
                    if epoch==0:
                        wr.writerow(key_name)

                    for k, vl in train_metrics.items():
                        train_list.append(vl)

                    train_list.append(val_loss)

                    for k, vl in val_metrics.items():
                        train_list.append(vl)
                    zip(train_list)
                    wr.writerow(train_list)


            except IOError:
                print("I/O Error")

            
            ##############################
            # Saving best model 
            ##############################
            
            if val_metrics['micro_f1']>=val_micro_f1_max:
                print('val micro f1 increased ({:.6f}-->{:.6f})'.format(val_micro_f1_max,val_metrics['micro_f1']))
                
                torch.save({'epoch':epoch+1,
                            'model_state_dict':model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'loss':val_loss},model_path+f'/C_micro_{epoch+1}_{batch_size}.pth')
                best_micro_model_path=model_path+f'/C_micro_{epoch+1}_{batch_size}.pth'
               
                val_micro_f1_max=val_metrics['micro_f1']

            if val_metrics['macro_f1'] >= val_macro_f1_max:
                print('val macro f1 increased ({:.6f}-->{:.6f})'.format(val_macro_f1_max,
                                                                                     val_metrics['macro_f1']))

                torch.save({'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'loss': val_loss}, model_path + f'/C_macro_{epoch + 1}_{batch_size}.pth')
                best_macro_model_path = model_path + f'/C_macro_{epoch + 1}_{batch_size}.pth'

                val_macro_f1_max = val_metrics['macro_f1']

            val_metrics['val_macro_f1_max'] = val_macro_f1_max
            val_metrics['val_micro_f1_max'] = val_micro_f1_max
            wandb.log(add_prefix(val_metrics, f'val'), step=epoch, commit=True)
            print('-'*10)




       
        time_elapsed=time.time()-since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
        training_curve(epochs,lossesT,lossesV)
        epochs.clear()
        lossesT.clear()
        lossesV.clear()

        best_model_paths = {"macro":best_macro_model_path,"micro":best_micro_model_path}
        test_metrics_dict = {"macro":None, "micro":None}
        ############################
        #         Test
        ############################
        for name,best_model_path in best_model_paths.items():
            test_list=[]
            print('Best model path:{}'.format(best_model_path))
            best_model=self.model
            checkpoint=torch.load(best_model_path,map_location=device)   # loading best model
            best_model.load_state_dict(checkpoint['model_state_dict'])
            best_model.to(device)
            best_model.eval()
            with torch.no_grad():
                   test_loss, test_metrics, test_num_steps=test_net(epoch,best_model,self.test_generator,device,criterion,args,self.center)
                #test_loss, test_metrics, test_num_steps=test_net(best_model,self.test_generator,device,criterion,args)



            print_metrics(test_metrics,test_num_steps)
            test_list.append(test_loss)


            for k, vl in test_metrics.items():
                test_list.append(vl)              # append metrics results in a list



            ##################################################################
            # Writing test results to a csv file
            ##################################################################

            key_name=['Test_loss','Test_micro_precision','Test_micro_recall','Test_micro_f1','Test_macro_precision','Test_macro_recall','Test_macro_f1','Test_mcc']
            try:

                    with open(filename+str('.csv'), 'a',newline="") as f:
                        wr = csv.writer(f,delimiter=",")
                        wr.writerow(key_name)
                        zip(test_list)
                        wr.writerow(test_list)
                        wr.writerow("")
            except IOError:
                    print("I/O Error")
            test_metrics_dict[name] = test_metrics
        return val_metrics, test_metrics_dict
        
        
                       
         
                
if __name__=="__main__":
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device: {device}')
    logging.info(f'''Starting training:
                 Epochs: {max_epochs}
                 Batch Size: {batch_size}
                 Learning Rate: {lr}''')

    reduce_ratio = args.data_reduce_rate * 100



    wandb_name = f"{args.model}_PreTrain_{args.pretrain_model}_{args.dataset}_"
    if args.data_reduce_rate < 1.0:
        reduce_ratio = args.data_reduce_rate * 100
        wandb_name += str(reduce_ratio) + "%_"
    if args.CE_loss_use:
        wandb_name += 'CE_'
        if args.CE_loss_weight:
            wandb_name += str(args.CE_loss_weight)+"_"
    if args.WCE_loss_use:
        wandb_name += 'WCE_'
        if args.WCE_loss_weight:
            wandb_name += str(args.WCE_loss_weight)+"_"
    if args.LDAM_loss_use:
        wandb_name += 'LDAM_'
        if args.LDAM_loss_weight:
            wandb_name += str(args.LDAM_loss_weight)+"_"
    if args.Logit_loss_use:
        wandb_name += 'Logit_'
        if args.Logit_loss_weight:
            wandb_name += str(args.Logit_loss_weight)+"_"
    if args.CCL_loss_use:
        wandb_name += 'CCL_'
        if args.CCL_loss_weight:
            wandb_name += str(args.CCL_loss_weight)+"_"
    if args.supcon_loss_use:
        wandb_name += 'SCL_'
        if args.supcon_loss_weight:
            wandb_name += str(args.supcon_loss_weight)
    wandb.init(dir=os.path.abspath("wandb"), project="Gastro",
               name=wandb_name
               , config=args, job_type='train', mode='online')
    t=train()
    t.train_net()
  




