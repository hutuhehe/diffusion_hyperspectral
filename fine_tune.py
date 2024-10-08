import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import optimizer
from torch.utils.data.dataloader import Sampler
from tqdm import tqdm
import json
import os
import gc

from torch.utils.data import DataLoader, WeightedRandomSampler

import argparse
from src.utils import setup_seed, multi_acc,FocalLoss
from src.pixel_classifier import  load_ensemble, compute_iou, predict_labels, save_predictions,calculate_metric_per_class_plot_cm, pixel_classifier, DualPathNetwork,Conv1D_Classfier
from src.datasets import ImageLabelDataset, FeatureDataset, make_transform
from src.feature_extractors import create_feature_extractor, collect_features
from src.cross_attention import DualPathNetwork_cross

from guided_diffusion.guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
from guided_diffusion.guided_diffusion.dist_util import dev
import random
import pdb

from peft import LoraConfig, get_peft_model

# Adopted from https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/train_interpreter.py#L434
def train(args):
   
    
    
    features, labels, data_spectral = prepare_data(args)

    train_data = FeatureDataset(features,labels,data_spectral)

    """
    suffix = '_'.join([str(step) for step in args['steps']])
    suffix += '_' + '_'.join([str(step) for step in opts['blocks']])+'.pt'
    filename = args['category']+"_"+ suffix
    train_data_file =  os.path.join(args['train_data_pt_folder'], filename)

    torch.save(train_data,train_data_file)
    
    pdb.set_trace()
    
    
    """
    
    #train_data = torch.load(args['train_data_pt_file'])

    
    
    print(f" ********* max_label {args['number_class']} *** ignore_label {args['ignore_label']} ***********")
    #print(f" *********************** Current number data {len(features)} ***********************")

    # modification on June 28, adding weighted sampler for imbalanced classes
    """
    
    # Create a WeightedRandomSampler
    
    class_weights = args['sample_weights']
    sample_weights = [0]*len(train_data)

    for idx,(_, label,_) in enumerate(train_data):
      class_weight = class_weights[label-1]
      sample_weights[idx] = class_weight 

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights),replacement = True)
    train_loader = DataLoader(dataset=train_data, batch_size=args['batch_size'], sampler= sampler)
    """
    
    train_loader = DataLoader(dataset=train_data, batch_size=args['batch_size'], shuffle=True, drop_last=True)
    print(" *********************** Current dataloader length " +  str(len(train_loader)) + " ***********************")
    
    for MODEL_NUMBER in range(args['start_model_num'], args['model_num'], 1):

        gc.collect()

        # Initialize networks
        #model = DualPathNetwork_cross(num_class =args['number_class']) #using cross attention dualnetwork, for berlin dataset
        #model = DualPathNetwork(num_class =args['number_class']) #using cross attention dualnetwork
        #model = DualPathNetwork(spatial_dim = args['dim'][-1],spectral_dim = args['bands_num'], num_class = args['number_class'])  # using naive feature concan
        model = pixel_classifier(numpy_class= args['number_class'], dim= args['dim'][-1])
        #model = Conv1D_Classfier(num_classes= args['number_class'],bands_num = args['bands_num'])



        # Load weights if using cross attention
        """
        state_dict = torch.load('model_0.pth')['model_state_dict']
        model.load_state_dict(state_dict)
        """

        model.to(dev())


        #weights = torch.ones(13)  # Start with weight of 1 for each class
        #weights  = torch.FloatTensor(args['sample_weights'])*10
        #criterion = nn.CrossEntropyLoss(weight=weights.to(dev()))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'],weight_decay=1e-4)
        iteration = 0
        break_count = 0
        best_loss = 10000000
        stop_sign = 0
        alpha = args['mix_up_alpha']

        for epoch in range(1000):
            model.train()
            total_loss = 0
            for X_spatial_batch, y_batch,X_spectral_batch in train_loader:

                X_spatial_batch, y_batch, X_spectral_batch = X_spatial_batch.to(dev()), y_batch.to(dev()),X_spectral_batch.to(dev())
                y_batch = y_batch.type(torch.long)
          
                # using mix up data for #
                #X_spatial_batch, X_spectral_batch, y_batch_mixed, lam = mixup_data(args,X_spatial_batch, X_spectral_batch, y_batch-1, alpha, device=dev())
                
                # Forward pass through the combined model
                #y_pred = model(X_spatial_batch, X_spectral_batch)  # for Dualnetwork, using both X_spatial and X_spectral
                y_pred = model(X_spatial_batch)  # for pixel classfier model, only using x_spatial
                #y_pred = model(X_spectral_batch)  # for conv1d classfier model, only using x_spectral

                #loss = mixup_criterion(y_pred, y_batch_mixed) # using this  loss if usingg mixup_data
                loss = criterion(y_pred, y_batch-1)
                optimizer.zero_grad()                      

                loss.backward()
                optimizer.step()
                
                total_loss+= loss.item()
  
                acc = multi_acc(y_pred, y_batch-1)
                iteration += 1
                if iteration % 1000 == 0:
                  print('Epoch : ', str(epoch), 'iteration', iteration, 'loss', loss.item(), 'acc', acc)
                
                if epoch > args['max_epoch'] : #change from 8 to 20 if using oversampling
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        break_count = 0
                    else:
                        break_count += 1

                    if break_count > 100:
                        stop_sign = 1
                        print("*************** Break, Total iters,", iteration, ", at epoch", str(epoch), "***************")
                        break

            if stop_sign == 1:
                break

        model_path = os.path.join(args['exp_dir'], 
                                  'model_' + str(MODEL_NUMBER) + '.pth')
        MODEL_NUMBER += 1
        print('save to:',model_path)
        torch.save({'model_state_dict': model.state_dict()},
                   model_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    pdb.set_trace()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())

    parser.add_argument('--exp', type=str)
    parser.add_argument('--seed', type=int,  default=0)

    args = parser.parse_args()
    setup_seed(args.seed)

    # Load the experiment config
    opts = json.load(open(args.exp, 'r'))
    opts.update(vars(args))
    opts['image_size'] = opts['dim'][0]

    # Check whether all models in ensemble are trained 
    pretrained = [os.path.exists(os.path.join(opts['exp_dir'], f'model_{i}.pth')) 
                  for i in range(opts['model_num'])]
              
    if not all(pretrained):
        # train all remaining models
        opts['start_model_num'] = sum(pretrained)
        train(opts)
    
    print('Loading pretrained models...')
    models = load_ensemble(opts, device='cuda')
    evaluation(opts, models)
