#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 15:37:10 2021

@author: yifeiren
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import json
import pickle
import datetime
import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score


import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

from tqdm import tqdm

from models import MultiPARAFAC2, TemporalDependency
from utils import EarlyStopping, AverageMeter, PaddedDenseTensor
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
import math

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.metrics import average_precision_score




use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

 
trainmimic = pickle.load( open ("extract3000.p", "rb") )
tensorlossrecord = []
tensorweightrecord = []
mortalweightrecord = []
epochadmiweight = []
epochicuweight = []
predictionlossrecord = []
epochadmissionloss = []
totallossrecord = []

def logistic_regression(Wtrain, y_train, Wtest, y_test):
    classifier = LogisticRegressionCV(cv=5, Cs=10, class_weight='balanced')
    classifier.fit(Wtrain, y_train)
    predictionloss = log_loss(y_train, classifier.predict_proba(Wtrain), eps=1e-15)
    pred_prob = classifier.predict_proba(Wtest)
    prauc = roc_auc_score(y_test, pred_prob[:, 1])
    return prauc,predictionloss


def predictionauc(W,label):
    best_auc = []
    largest_loss = []
    for i in range(5):
        Wtrain, Wtest, labels_train, labels_test = train_test_split(W, label, 
                                                                    train_size=0.8, test_size=0.2)
        prauc,predictionloss = logistic_regression(Wtrain, labels_train, Wtest, labels_test)
        best_auc.append(prauc)
        largest_loss.append(predictionloss)

            
        
    return max(best_auc),max(largest_loss)


def lstmauc(U,ventilabel):
    labeldata = ventilabel
    UU = U.data.numpy()
    train_size = int(len(UU) * 0.8)
    #test_size = len(UU) - train_size
    train, test = UU[0:train_size,:], UU[train_size:len(U),:]
    trainY, testY = labeldata[0:train_size],labeldata[train_size:len(U)]
    trainY = np.array(trainY)
    testY = np.array(testY)
    model = Sequential()
    model.add(LSTM(4))
    model.add(Dense(240))
    model.compile(loss='mean_squared_error', optimizer='adam')
    hist = model.fit(train, trainY, epochs=1, batch_size=1, verbose=2)
    testPredict = model.predict(test)
    lstmloss = hist.history['loss']
    aucsum = 0
    for i in range(5):
        prauc = roc_auc_score(testY[:,i], testPredict[:,i])
        aucsum = aucsum + prauc
    aucsum = aucsum/5

    
    return aucsum, lstmloss[0]



def validatesquarederror(model, dataloader):
    with torch.no_grad():
        targets = []
        predictions = []
        for pids, Xdense, masks in dataloader:
            pids, Xdense, masks = pids.to(device), Xdense.to(device), masks.to(device)
            output = model(pids)
            output = output
            target = Xdense
            targets.append(target)
            predictions.append(output)
        targets = torch.cat(targets, dim=0)
        predictions = torch.cat(predictions, dim=0)
        normX = sum(sum(sum(targets**2)))
        me = sum(sum(sum((targets-predictions)**2)))

        fit =1-(me/normX)
    return fit

def train_mul_parafac2(indata, 
                            num_visits, 
                            num_feats, 
                            log_path, 
                            pos_prior, 
                            reg_weight, 
                            smooth_weight,
                            rank, 
                            weight_decay, 
                            alpha, 
                            gamma, 
                            lr, 
                            seed, 
                            batch_size, 
                            smooth_shape,
                            iters, 
                            patience, 
                            label,
                            readmissionlabel,
                            iculabel,
                            ventilabel):

    if seed is not None:
        torch.manual_seed(seed)

    model = MultiPARAFAC2(num_visits, 
                             num_feats, 
                             rank, 
                             alpha=alpha, 
                             gamma=gamma).to(device)

    temp_model = TemporalDependency(rank=50,nlayers=1, nhidden=100, dropout=0)

    tf_loss_func = nn.MSELoss(reduction = 'sum')
 

    optimizer_pt_U = Adam([model.U],lr=lr, weight_decay=weight_decay)
    optimizer_pt_S = Adam([model.S],lr=lr, weight_decay=weight_decay)
    optimizer_pt_V = Adam([model.V], lr=lr, weight_decay=weight_decay)
    

    lr_scheduler_pt_U = ReduceLROnPlateau(optimizer_pt_U , 
                                             mode='max', 
                                             cooldown=10, 
                                             min_lr=1e-6)
    lr_scheduler_pt_S = ReduceLROnPlateau(optimizer_pt_S , 
                                             mode='max', 
                                             cooldown=10, 
                                             min_lr=1e-6)
    lr_scheduler_pt_V = ReduceLROnPlateau(optimizer_pt_V, 
                                                mode='max', 
                                                cooldown=10, 
                                                min_lr=1e-6)
    

    collators = [PaddedDenseTensor(indata, num_feats, subset=subset) 
                 for subset in ('train', 'validation', 'test')]
    loaders = [DataLoader(TensorDataset(torch.arange(len(num_visits))), 
                          shuffle=True, 
                          num_workers=2, 
                          batch_size=batch_size, 
                          collate_fn=collator)
               for collator in collators]
    train_loader, valid_loader, test_loader = loaders

    early_stopping = EarlyStopping(patience=patience)

    aucscore = 0
    predictionloss1 = 1
    admissionloss1 = 1
    iculoss1 = 1
    ventiloss1 = 1
    tensorloss1 = 1
    predictionloss2 = 0
    admissionloss2 = 0
    iculoss2 = 0
    ventiloss2 = 0
    tensorloss2 = 0
    
    predictionloss = 1
    admissionloss = 1
    iculoss = 1
    ventiloss = 1
    tensorloss = 1
    
    T = 0.3
    venti_weight = 1
    mortal_weight = 1
    admi_weight = 1
    icu_weight = 1
    tensor_weight = 1
    
    tensorlosstotal = 0

    epochcount = 0
    epochtensorloss = 0

    predictionlosstotal = 0
    admissionlosstotal = 0
    iculosstotal = 0
    ventilosstotal = 0
    for epoch in range(iters):
        epoch_tf_loss = AverageMeter()
        epoch_uni_reg = AverageMeter()

        pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}')
        lr = optimizer_pt_U.param_groups[0]['lr']

        itrcount = 0
        for pids, Xdense, masks  in train_loader:
        
        
            num_visits_batch = masks.squeeze(-1).sum(dim=1).to(device)

            num_visits_batch, pt_idx = num_visits_batch.sort(descending=True)
            pids = pids[pt_idx].to(device)
            Xdense = Xdense[pt_idx].to(device)
            masks = masks[pt_idx].to(device)



            # update U 
            model.S.requires_grad = False
            model.U.requires_grad = True
            model.V.requires_grad = False
            
            optimizer_pt_U.zero_grad()
            
            output = model(pids)
            out = tf_loss_func(output, Xdense)
           
            uni_reg = model.uniqueness_regularization(pids)
            out = out + reg_weight * uni_reg


            out.backward()
            optimizer_pt_U.step()
            model.projection()
            epoch_uni_reg.update(uni_reg.item(), n=pids.shape[0])
            
            # update S
            
            model.S.requires_grad = True
            model.U.requires_grad = False
            model.V.requires_grad = False
            
            optimizer_pt_S.zero_grad()
            
            output = model(pids)

            tfloss = tf_loss_func(output, Xdense)
            tensorloss = tensor_weight * tfloss
            tensorlosstotal = tensorlosstotal + tfloss
            uni_reg = model.uniqueness_regularization(pids)
            out = tensorloss + reg_weight * uni_reg

            out = out + mortal_weight * predictionloss
          


            
            out.backward()
            optimizer_pt_S.step()
            model.projection()
            epoch_uni_reg.update(uni_reg.item(), n=pids.shape[0])
            
            

            # update V
            model.S.requires_grad = False
            model.U.requires_grad = False
            model.V.requires_grad = True
            optimizer_pt_V.zero_grad()
            output = model(pids)
            out = tf_loss_func(output, Xdense)
            out.backward()
            optimizer_pt_V.step()
            model.projection()

            pbar.update()
            
            itrcount = itrcount + 1
        
        epochcount = epochcount + 1
        avetensorloss = tensorlosstotal/itrcount  

        
        model.update_phi()
        for i, num_visit in enumerate(num_visits):
            model.U.data[i, num_visit:] = 0
        W = np.array(model.S)
        
        aucscoref,predictionloss = predictionauc(W,label)
        aucscorefadmission,admissionloss = predictionauc(W,readmissionlabel)
        aucscoreficu,iculoss = predictionauc(W,iculabel)
        aucscoreventi,ventiloss = lstmauc(model.U,ventilabel)
        print(f'morhosauc: {aucscoref:.3f}')
        print(f'admissionauc: {aucscorefadmission:.3f}')
        print(f'icuauc: {aucscoreficu:.3f}')
        print(f'ventiauc: {aucscoreventi:.3f}')
        predictionlosstotal = predictionlosstotal + predictionloss
        admissionlosstotal = admissionlosstotal + admissionloss
        iculosstotal = iculosstotal + iculoss
        ventilosstotal = ventilosstotal + ventiloss
            
            
            
        epochtensorloss = epochtensorloss + avetensorloss 

        tensorlosstotal = 0

        if epochcount % 5 == 0 :

        
            predictionlosssmooth = predictionlosstotal / epochcount
            tensorlosssmooth = epochtensorloss /epochcount
            admissionlosssmooth = admissionlosstotal/epochcount
            iculosssmooth = iculosstotal/epochcount
            ventilosssmooth = ventilosstotal/epochcount
            weightedtotallosssmooth = tensor_weight * tensorlosssmooth + mortal_weight * predictionlosssmooth + admi_weight * admissionlosssmooth + icu_weight * iculosssmooth + venti_weight * ventilosssmooth 

            predictionloss2 = predictionlosssmooth
            admissionloss2 = admissionlosssmooth
            iculoss2 = iculosssmooth
            tensorloss2 = tensorlosssmooth
            ventiloss2 = ventilosssmooth
        

        

            if epochcount == 5 :
                predictionloss1 = predictionloss2
                admissionloss1 = admissionloss2
                iculoss1 = iculoss2
                ventiloss1 = ventiloss2
                tensorloss1 = tensorloss2
            
            
            interweightprediction = predictionloss2/(predictionloss1*T)
            #print(f' interweightprediction: { interweightprediction:.3f}')
            interweightpredictionexp = math.exp(interweightprediction)
            interweightadmission = admissionloss2/(admissionloss1*T)
            ##print(f'interweightadmission: {interweightadmission:.3f}')
            interweightadmissionexp = math.exp(interweightadmission)
            interweighticu = iculoss2/(iculoss1*T)
            ##print(f'interweighticu: {interweighticu:.3f}')
            interweighticuexp = math.exp(interweighticu)
            interweightventi = ventiloss2/(ventiloss1*T)
            ##print(f'interweighticu: {interweighticu:.3f}')
            interweightventiexp = math.exp(interweightventi)
            interweighttensor = tensorloss2/(tensorloss1*T)
            #print(f'interweighttensor: {interweighttensor:.3f}')
            interweighttensorexp = math.exp(interweighttensor)
        


       
            totalexp = interweightpredictionexp + interweightadmissionexp + interweighticuexp + interweighttensorexp + interweightventiexp
    
            venti_weight = 5 * interweightventiexp/totalexp
            mortal_weight = 5 * interweightpredictionexp/totalexp
            admi_weight = 5 * interweightadmissionexp/totalexp
            icu_weight = 5 * interweighticuexp/totalexp
            tensor_weight = 5 * interweighttensorexp/totalexp
        
            print(f'tensor weight: {tensor_weight:.3f}')
            print(f'mortality weight: {mortal_weight:.3f}')
            print(f'admission weight: {admi_weight:.3f}')
            print(f'icu weight: {icu_weight:.3f}')

        
            fit = validatesquarederror(model, valid_loader)
            print(f'FIT: {fit:.3f}')
            print(f'lr: {lr:.3f}')
            lr_scheduler_pt_U.step(fit)
            lr_scheduler_pt_S.step(fit)
            lr_scheduler_pt_V.step(fit)
        
            predictionloss1 = predictionloss2
            admissionloss1 = admissionloss2
            iculoss1 = iculoss2
            ventiloss1 = ventiloss2
            tensorloss1 = tensorloss2
   
        
            tensorlossrecord.append(tensorlosssmooth.item())
            predictionlossrecord.append(predictionlosssmooth)
            totallossrecord.append(weightedtotallosssmooth.item())
            tensorweightrecord.append(tensor_weight)
            mortalweightrecord.append(mortal_weight)

 
            
            epochtensorloss = 0
            epochcount = 0
            predictionlosstotal = 0
            admissionlosstotal = 0
            iculosstotal = 0
            ventilosstotal = 0

        
    return model.U, model.S, model.V,model




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str,
                        help='Name of the experiment.')
    parser.add_argument('--data_path', '-d', type=str, 
                        default='./demo_data.pkl',
                        help='The path of input data.')
    parser.add_argument('--pi', '-p', type=float, default=0.005,
                        help='Class prior for the positive observations.')
    parser.add_argument('--uniqueness', '-u', type=float, default=1e-3,
                        help='Weighting for the uniqueness regularization.')
    parser.add_argument('--rank', '-r', type=int, default=50,
                        help='Target rank of the PARAFAC2 factorization.')
    parser.add_argument('--seed', type=int, 
                        help='Random seed')
    parser.add_argument('--alpha', type=float, default=2,
                        help='Maximam infinity norm allowed for the factor '\
                             'matrices.')
    parser.add_argument('--gamma', type=float, default=1,
                        help='Shape parameter for the sigmoid function.')
    parser.add_argument('--lr', type=float, default=1e-1,
                        help='Learning rate of the optimizers.')
    parser.add_argument('--wd', type=float, default=0,
                        help='Weight decay for the optimizers.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--proj_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20,
                        help='Epochs to wait before early stopping, use 0 to '\
                             'switch off early stopping.')
    parser.add_argument('--smooth', type=float, default=1,
                        help='Weighting for the smoothness regularization.')
    parser.add_argument('--smooth_shape', type=float, default=1,
                        help='shape parameter for the time-aware TV smoothing')
    parser.add_argument('--out', type=str, default='./results/',
                        help='Directory to save the results, a subfolder will '\
                             'be created.')
    parser.add_argument('--log', type=str, default='./results/tb_logs',
                        help='Path to store the tensorboard logging file.')

    args = parser.parse_args()

        
    ##########################
    ## Load data #############
    ##########################
    indata = trainmimic
    data_train = indata

    num_feats = max([pt['train'][:, 1].max() + 1 for pt in indata])
        

    num_visits_train = [pt['times'] for pt in data_train]

    


    ##################################
    ## Set up experiment #############
    ##################################
    exp_id = f'LogPar_rank{args.rank}'
    if args.name is not None:
        exp_id = args.name + '_' + exp_id
    if args.seed is not None:
        exp_id += f'_seed{args.seed}'
    exp_id += f'_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
    results_out_dir = Path(args.out) / exp_id

    results_out_dir.mkdir(parents=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    with open(results_out_dir/'config.json', 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    ###############################
    ## Run experiment #############
    ###############################
    print(f'rank={args.rank}, seed={args.seed}')
    labeldata = []
    readmissionlabeldata = []
    iculabeldata = []
    ventilabeldata = []
    for i in range(3000):
        labeldata.append(data_train[i]["mortalityhosp"])
        readmissionlabeldata.append(data_train[i]["readmission"])
        iculabeldata.append(data_train[i]["mortalityicu"])
        ventilabeldata.append(data_train[i]["ventilation"])
    U,S,V,model = train_mul_parafac2(
        data_train, num_visits_train, int(num_feats), 
        patience=args.patience or args.epochs, 
        log_path=os.path.join(args.log, exp_id),
        pos_prior=args.pi,
        smooth_weight=args.smooth,
        seed=args.seed,
        rank=args.rank,
        reg_weight=args.uniqueness,
        weight_decay=args.wd,
        alpha=args.alpha,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        smooth_shape=args.smooth_shape,
        iters=args.epochs,
        label=labeldata,
        readmissionlabel = readmissionlabeldata,
        iculabel = iculabeldata,
        ventilabel = ventilabeldata)

    W = np.array(S)
    #import pickle 

    #pickle.dump(V, open( "V.p", "wb" ) ) 
    
