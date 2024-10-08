import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import optimizer
from tqdm import tqdm
import json
import os
import gc

from torch.utils.data import DataLoader

import argparse
from src.utils import setup_seed, multi_acc
from src.pixel_classifier import  load_ensemble, compute_iou, predict_labels, save_predictions, save_predictions,calculate_metric_per_class, pixel_classifier, DualPathNetwork,Conv1D_Classfier
from src.datasets import ImageLabelDataset, FeatureDataset, make_transform
from src.feature_extractors import create_feature_extractor, collect_features

from guided_diffusion.guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
from guided_diffusion.guided_diffusion.dist_util import dev
import pdb


def prepare_data(args):
    feature_extractor = create_feature_extractor(**args)
    #pdb.set_trace()
    print(f"Preparing the train set for {args['category']}...")
    dataset = ImageLabelDataset(
        data_dir=args['training_path'],
        resolution=args['image_size'],
        num_images=args['training_number'],
        transform=make_transform(
            args['model_type'],
            args['image_size']
        )
    )
    
    #X = torch.zeros((len(dataset), *args['dim'][::-1]), dtype=torch.float)
    #y = torch.zeros((len(dataset), *args['dim'][:-1]), dtype=torch.uint8)
    
    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], 
                            generator=rnd_gen, device=dev())
    else:
        noise = None 
    
    pixel_num  = torch.tensor(0)
    #d  = args['dim'][-1]
    for row, (img, label,data_patch) in enumerate(tqdm(dataset)):
        #pdb.set_trace()
        img = img[None].to(dev())
        features = feature_extractor(img, noise=noise)
        # read spectral data from data_patch
        X_spectral = data_patch  #X_spectral shape[144,64,64]
        # resize feature maps
        X_spatial = collect_features(args, features).cpu() 
        #X[row] = collect_features(args, features).cpu()
        assert X_spatial.shape[0] == args['dim'][-1], "Dimension mismatch: X.shape[1] is not equal to args['dim'][2]"
        """
        for target in range(args['number_class']):
            #pdb.set_trace()
            if target == args['ignore_label']: continue
            
            if 0 < (label == target).sum() < 20:
                print(f'Delete small annotation from image {dataset.image_paths[row]} | label {target}')
                label[label == target] = args['ignore_label']
        """
            
        #y[row] = label
        y = label
        
        X_spatial = X_spatial.reshape(args['dim'][-1],-1).permute(1,0)
        X_spectral = X_spectral.reshape(args['bands_num'],-1).permute(1,0)
        # Compute the L2 norm for each sample
        #norms = torch.norm(X_spectral, p=2, dim=1, keepdim=True)  # keepdim=True to keep the dimension for broadcasting
        # Normalize each sample
        #epsilon = 1e-10  # Small number to avoid division by zero
        # X_spectral = X_spectral /65535

        y = y.flatten()
        
        X_spatial = X_spatial[y != args['ignore_label']]
        X_spectral= X_spectral[y != args['ignore_label']]
        y = y[y != args['ignore_label']]
        
        #Concantenate the X and y
        if row == 0:
          concan_X_spatial = X_spatial
          concan_X_spectral = X_spectral
          concan_y = y
          
        else:
          concan_X_spatial = torch.cat((concan_X_spatial, X_spatial), dim  = 0)
          concan_X_spectral = torch.cat((concan_X_spectral, X_spectral), dim  = 0)
          concan_y = torch.cat((concan_y, y),dim  =0 )
        """
        for i in range(X.shape[0]):
          file_path  = os.path.join(args['feature_save_path'],f'feature_{i + pixel_num}_{y[i]}.npy')
          np.save(file_path, X[i].cpu().numpy())
  
        pixel_num  += X.shape[0]
        """
    
    #d = X.shape[1]
    #print(f'Total dimension {d}')
    #pdb.set_trace()
  
    return concan_X_spatial, concan_y, concan_X_spectral
    #return X[y != args['ignore_label']], y[y != args['ignore_label']]


def evaluation(args, models):
    feature_extractor = create_feature_extractor(**args)
    dataset = ImageLabelDataset(
        data_dir=args['testing_path'],
        resolution=args['image_size'],
        num_images=args['testing_number'],
        transform=make_transform(
            args['model_type'],
            args['image_size']
        )
    )

    #pdb.set_trace()
    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], 
                            generator=rnd_gen, device=dev())
    else:
        noise = None 

    preds, gts, uncertainty_scores = [], [], []
    for img, label,data_patch in tqdm(dataset):        
        #pdb.set_trace()
        img = img[None].to(dev())
        features = feature_extractor(img, noise=noise)
        features = collect_features(args, features)

        x_spatial= features.view(args['dim'][-1], -1).permute(1, 0)
        x_spectral = data_patch.reshape(args['bands_num'],-1).permute(1,0)

        pred, uncertainty_score = predict_labels(
            models, x_spatial,x_spectral, size=args['dim'][:-1]
        )
        gts.append(label.numpy()-1)
        preds.append(pred.numpy())
        uncertainty_scores.append(uncertainty_score.item())
    
    inference_map= save_predictions(args, dataset.image_paths, preds)
    test_label = np.load('test_label.npy')
    miou, accuracy, _, _ = calculate_metric_per_class(args,test_label, inference_map, args['number_class'])
    #miou = compute_iou(args, preds, gts)
    print(f'Overall mIoU: {miou}')
    print(f'Overall accuracy: {accuracy}')
    print(f'Mean uncertainty: {sum(uncertainty_scores) / len(uncertainty_scores)}')



def mixup_data(x_spatial, x_spectral, y, alpha=0.2, device='cpu'):

    if alpha > 0:
        lam_spatial = np.random.beta(alpha, alpha)
        lam_spectral = np.random.beta(alpha, alpha)
    else:
        lam_spatial = 1
        lam_spectral = 1
    lam = (lam_spatial+lam_spectral)/2


    batch_size = x_spatial.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x_spatial = lam_spatial * x_spatial + (1 - lam_spatial) * x_spatial[index, :]
    mixed_x_spectral = lam_spectral * x_spectral + (1 - lam_spectral) * x_spectral[index, :]

    y_one_hot = torch.nn.functional.one_hot(y, num_classes=15)  # Assume num_classes is the second dimension
    mixed_y = lam * y_one_hot.float() + (1 - lam) * y_one_hot[index, :].float()

    return mixed_x_spatial, mixed_x_spectral, mixed_y, lam

def mixup_criterion(pred, mixed_y):
    '''Computes the cross entropy loss for mixed targets.
    Args:
        pred (Tensor): Predictions from the model.
        mixed_y (Tensor): Mixed labels, in one-hot format.
    Returns:
        loss (Tensor): The computed cross entropy loss.
    '''
    log_softmax = torch.nn.LogSoftmax(dim=1)(pred)
    loss = -torch.sum(mixed_y * log_softmax, dim=1)
    return torch.mean(loss)



# Adopted from https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/train_interpreter.py#L434
def train(args):

    """
    features, labels, data_spectral = prepare_data(args)

    train_data = FeatureDataset(features,labels,data_spectral)
    
    torch.save(train_data,'train_data.pt')
    """
    
    train_data = torch.load('train_data.pt')
    pdb.set_trace()
    
    print(f" ********* max_label {args['number_class']} *** ignore_label {args['ignore_label']} ***********")
    #print(f" *********************** Current number data {len(features)} ***********************")

    train_loader = DataLoader(dataset=train_data, batch_size=args['batch_size'], shuffle=True, drop_last=True)
    print(" *********************** Current dataloader length " +  str(len(train_loader)) + " ***********************")
    
    for MODEL_NUMBER in range(args['start_model_num'], args['model_num'], 1):

        gc.collect()
        # Initialize networks
        model = DualPathNetwork(num_class =args['number_class']) 
        #model = DualPathNetwork(spatial_dim = args['dim'][-1],spectral_dim =144, num_class = 15)
        #model = pixel_classifier(numpy_class= 15, dim= args['dim'][-1])
        #model = Conv1D_Classfier(num_classes= 15)
        model.to(dev())

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-5)

        iteration = 0
        break_count = 0
        best_loss = 10000000
        stop_sign = 0
        alpha = 0.1


        for epoch in range(100):
            model.train()
            total_loss = 0
            for X_spatial_batch, y_batch,X_spectral_batch in train_loader:
                X_spatial_batch, y_batch, X_spectral_batch = X_spatial_batch.to(dev()), y_batch.to(dev()),X_spectral_batch.to(dev())
                y_batch = y_batch.type(torch.long)
                #pdb.set_trace()
                X_spatial_batch, X_spectral_batch, y_batch_mixed, lam = mixup_data(X_spatial_batch, X_spectral_batch, y_batch-1, alpha=0.2, device=dev())
                
                # Forward pass through the combined model
                y_pred = model(X_spatial_batch, X_spectral_batch)
                
                #y_pred = model(X_spatial_batch)
                #y_pred = model(X_spectral_batch)

                loss = mixup_criterion(y_pred, y_batch_mixed)
                #loss = criterion(y_pred, y_batch-1)
                optimizer.zero_grad()                      

                loss.backward()
                optimizer.step()
                
                total_loss+= loss.item()
  
                acc = multi_acc(y_pred, y_batch-1)
                iteration += 1
                if iteration % 200 == 0:
                  print('Epoch : ', str(epoch), 'iteration', iteration, 'loss', loss.item(), 'acc', acc)
                
                if epoch > 8:
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        break_count = 0
                    else:
                        break_count += 1

                    if break_count > 50:
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

    # Prepare the experiment folder 
    if len(opts['steps']) > 0:
        suffix = '_'.join([str(step) for step in opts['steps']])
        suffix += '_' + '_'.join([str(step) for step in opts['blocks']])
        opts['exp_dir'] = os.path.join(opts['exp_dir'], suffix)

    path = opts['exp_dir']
    os.makedirs(path, exist_ok=True)
    print('Experiment folder: %s' % (path))
    os.system('cp %s %s' % (args.exp, opts['exp_dir']))

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
