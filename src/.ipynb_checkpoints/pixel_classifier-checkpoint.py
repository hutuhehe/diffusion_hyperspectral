import os
import torch
import torch.nn as nn
import numpy as np
from collections import Counter


from torch.distributions import Categorical
from src.utils import colorize_mask, oht_to_scalar
from src.data_util import get_palette, get_class_names
from src.cross_attention import DualPathNetwork_cross

from PIL import Image
from collections import defaultdict
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import seaborn as sns


"""
class MultiheadCrossAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(MultiheadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # Ensure the hidden dimension is divisible by the number of heads
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by the number of heads."

        self.query_transform = nn.Linear(input_dim, hidden_dim)
        self.key_transform = nn.Linear(input_dim, hidden_dim)
        self.value_transform = nn.Linear(input_dim, hidden_dim)
        self.out_transform = nn.Linear(hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, concat_spatial_spectral):
        batch_size = query.size(0)

        # Transform inputs
        query = self.norm(self.query_transform(query))  # (batch_size, seq_length, hidden_dim)
        keys = self.norm(self.key_transform(concat_spatial_spectral))
        values = self.norm(self.value_transform(concat_spatial_spectral))

        # Split into multiple heads
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        keys = keys.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(query, keys.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, seq_length, seq_length)
        attended_values = torch.matmul(attention_weights, values)  # (batch_size, num_heads, seq_length, head_dim)

        # Concatenate heads and project
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.out_transform(attended_values)  # (batch_size, seq_length, hidden_dim)

        # Optional residual connection
        output += query.view(batch_size, -1, self.num_heads * self.head_dim)

        return output

class DualPathNetwork(nn.Module):
    # cross attention modality fusion with multihead attention
    def __init__(self, num_class, hidden_dim=128, spatial_dim=8064, num_heads=4):
        super(DualPathNetwork, self).__init__()
        self.out_channels = 256
        self.spatial_process = nn.Sequential(
            nn.Linear(spatial_dim, self.out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.out_channels)
        )

        self.spectral_sequence = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, self.out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(num_features=self.out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.cls_token_spectral = nn.Parameter(torch.randn(1, 1, self.out_channels))
        self.pos_embedding_spectral = nn.Parameter(torch.randn(36 + 1, self.out_channels) * 0.01)

        self.spatial_attention = MultiheadCrossAttention(self.out_channels, hidden_dim, num_heads)
        self.spectral_attention = MultiheadCrossAttention(self.out_channels, hidden_dim, num_heads)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.classifier = nn.Linear(hidden_dim, num_class)

    def forward(self, X_spatial, X_spectral):
        batch_size = X_spectral.size(0)
        spectral_sequence = self.spectral_sequence(X_spectral.unsqueeze(1))
        X_spectral = spectral_sequence.transpose(1, 2)

        X_spatial = self.spatial_process(X_spatial)
        X_spatial = X_spatial.unsqueeze(1)

        cls_token_spectral = self.cls_token_spectral.expand(batch_size, -1, -1)
        X_spectral = torch.cat((cls_token


class CrossAttention(nn.Module):
    def __init__(self, input_dim , hidden_dim):
        super(CrossAttention, self).__init__()

        self.out_channels =  256
        self.norm = nn.LayerNorm(hidden_dim)

        self.query_transform = nn.Linear(self.out_channels, hidden_dim)
        self.key_transform = nn.Linear(self.out_channels, hidden_dim)
        self.value_transform = nn.Linear(self.out_channels, hidden_dim)

        self.scale = torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
        self.norm=  nn.LayerNorm(hidden_dim)

    def forward(self, query, concat_spatial_spectral):


        query_no_norm = self.query_transform(query)
        query = self.norm(query_no_norm)

        keys = self.key_transform(concat_spatial_spectral) # (batch_size, seq_legth, hidden_dim)
        keys = self.norm(keys)

        values = self.value_transform(concat_spatial_spectral)
        values = self.norm(values)


        attention_scores = torch.bmm(query, keys.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_output = torch.bmm(attention_weights, values)

        output_with_residual = attended_output + query_no_norm # (batch_size, hidden_dim)

        return output_with_residual



# build cross attention class using builit in multihead June/17
class CrossAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(CrossAttention, self).__init__()

        # Initialize the multi-head attention layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

        # Linear transformations for query, key, and value
        self.query_transform = nn.Linear(input_dim, hidden_dim)
        self.key_value_transform = nn.Linear(input_dim, hidden_dim)

        # Output transformation
        self.out_transform = nn.Linear(hidden_dim, input_dim)

        # Layer normalization
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, query, concat_spatial_spectral):
        # Transform query, key, and value
        query = self.query_transform(query).permute(1, 0, 2)  # [seq_len, batch_size, hidden_dim]
        key_value = self.key_value_transform(concat_spatial_spectral).permute(1, 0, 2)  # [seq_len, batch_size, hidden_dim]

        # Apply multi-head attention
        attn_output, _ = self.multihead_attn(query, key_value, key_value)

        # Apply the output transformation
        output = self.out_transform(attn_output.permute(1, 0, 2))  # Convert back to [batch_size, seq_len, hidden_dim]

        # Add residual connection and normalize
        output_with_residual = self.norm(output + query.permute(1, 0, 2))

        return output_with_residual


class DualPathNetwork(nn.Module):
    # cross attention modality fusion ,neweset version for Berlin dataset June/17
    def __init__(self, num_class, hidden_dim  = 128, spatial_dim = 8064):
        super(DualPathNetwork, self).__init__()
        self.out_channels = 256
        self.spatial_process = nn.Sequential(
            nn.Linear(spatial_dim, self.out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(num_features= self.out_channels))

        self.spectral_sequence = nn.Sequential(
            #nn.Conv1d(1, 32, kernel_size=3, padding=1,stride = 1), # If UH dataset using kernel_size = 3
            nn.Conv1d(1, 32, kernel_size=7, padding=3,stride = 1), # If Berlin dataset using kernel_size = 7
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, self.out_channels, kernel_size=3, padding=1,stride = 1),
            nn.BatchNorm1d(num_features =self.out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
            )

        self.cls_token_spectral = nn.Parameter(torch.randn(1, 1, self.out_channels))
        #self.pos_embedding_spectral = nn.Parameter(torch.randn(36 + 1, self.out_channels)*0.01) # 36 if using UH dataset
        self.pos_embedding_spectral = nn.Parameter(torch.randn(61 + 1, self.out_channels)*0.01) 
        self.spatial_attention = CrossAttention(self.out_channels, hidden_dim)
        self.spectral_attention = CrossAttention(self.out_channels, hidden_dim)

        # Trainable weights for combining outputs
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.classifier = nn.Linear(hidden_dim, num_class)


    def forward(self, X_spatial, X_spectral):
        batch_size = X_spectral.size(0)

        # Transform the input
        spectral_sequence = self.spectral_sequence(X_spectral.unsqueeze(1)) #(batch_size,out_channels,seq_length)
        X_spectral = spectral_sequence.transpose(1,2)  #(batch_size, seq_length, out_channels)

        X_spatial = self.spatial_process(X_spatial) #(batch_size,  out_channels )
        X_spatial = X_spatial.unsqueeze(1) #(batch_size,1, out_channels)

        cls_token_spectral = self.cls_token_spectral.expand(batch_size,-1, -1)
        X_spectral = torch.cat((cls_token_spectral, X_spectral), dim=1) + self.pos_embedding_spectral.unsqueeze(0)

        concate_spatial_spectral = torch.cat((X_spatial,X_spectral),dim =1)  #(batch_size, seg_len, out_channels)


        spatial_attended = self.spatial_attention(X_spatial, concate_spatial_spectral)
        spectral_attended = self.spectral_attention(X_spectral, concate_spatial_spectral)

        # Normalize and combine weights
        weights = F.softmax(self.weights, dim=0)
        spatial_logits = self.classifier(spatial_attended.squeeze(1))
        spectral_logits = self.classifier(spectral_attended[:,0,:])
        logits  = weights[0] * spatial_logits + weights[1] * spectral_logits
        #logits = self.classifier(combined_output.mean(dim=1))  # Assume average pooling over sequence
        return logits



class DualPathNetwork(nn.Module):
    # cross attention modality fusion ,neweset version for uh 
    def __init__(self, num_class, hidden_dim  = 128, spatial_dim = 8064):
        super(DualPathNetwork, self).__init__()
        self.out_channels = 256
        self.spatial_process = nn.Sequential(
            nn.Linear(spatial_dim, self.out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(num_features= self.out_channels))

        self.spectral_sequence = nn.Sequential(
            #nn.Conv1d(1, 32, kernel_size=3, padding=1,stride = 1), # If UH dataset using kernel_size = 3
            nn.Conv1d(1, 32, kernel_size=7, padding=3,stride = 1), # If Berlin dataset using kernel_size = 7
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, self.out_channels, kernel_size=3, padding=1,stride = 1),
            nn.BatchNorm1d(num_features =self.out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
            )

        self.cls_token_spectral = nn.Parameter(torch.randn(1, 1, self.out_channels))
        #self.pos_embedding_spectral = nn.Parameter(torch.randn(36 + 1, self.out_channels)*0.01) # 36 if using UH dataset
        self.pos_embedding_spectral = nn.Parameter(torch.randn(61 + 1, self.out_channels)*0.01) 
        self.spatial_attention = CrossAttention(self.out_channels, hidden_dim)
        self.spectral_attention = CrossAttention(self.out_channels, hidden_dim)

        # Trainable weights for combining outputs
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.classifier = nn.Linear(hidden_dim, num_class)


    def forward(self, X_spatial, X_spectral):
        batch_size = X_spectral.size(0)

        # Transform the input
        spectral_sequence = self.spectral_sequence(X_spectral.unsqueeze(1)) #(batch_size,out_channels,seq_length)
        X_spectral = spectral_sequence.transpose(1,2)  #(batch_size, seq_length, out_channels)

        X_spatial = self.spatial_process(X_spatial) #(batch_size,  out_channels )
        X_spatial = X_spatial.unsqueeze(1) #(batch_size,1, out_channels)

        cls_token_spectral = self.cls_token_spectral.expand(batch_size,-1, -1)
        X_spectral = torch.cat((cls_token_spectral, X_spectral), dim=1) + self.pos_embedding_spectral.unsqueeze(0)

        concate_spatial_spectral = torch.cat((X_spatial,X_spectral),dim =1)  #(batch_size, seg_len, out_channels)


        spatial_attended = self.spatial_attention(X_spatial, concate_spatial_spectral)
        spectral_attended = self.spectral_attention(X_spectral, concate_spatial_spectral)

        # Normalize and combine weights
        weights = F.softmax(self.weights, dim=0)
        spatial_logits = self.classifier(spatial_attended.squeeze(1))
        spectral_logits = self.classifier(spectral_attended[:,0,:])
        logits  = weights[0] * spatial_logits + weights[1] * spectral_logits
        #logits = self.classifier(combined_output.mean(dim=1))  # Assume average pooling over sequence
        return logits

"""

"""

class DualPathNetwork(nn.Module):
    # cross attention modality fusion
    def __init__(self, num_class, d_model = 256, nhead=1, num_layers = 0 , seq_len_spatial= 3, seq_len_spectral= 36, feature_dim_spatial= 2688, feature_dim_spectral= 32):
        super(DualPathNetwork, self).__init__()

          self.out_channels = 256
          self.spatial_process = nn.Sequential(   
            nn.Linear(spatial_dim, self.out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(num_features= self.out_channels) 
        )

        self.spectral_sequence = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features =self.out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2))

        self.spatial_attention = CrossAttention(spatial_dim, spatial_dim + spectral_dim, hidden_dim)
        self.spectral_attention = CrossAttention(spectral_dim, spatial_dim + spectral_dim, hidden_dim)

        # Trainable weights for combining outputs
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.classifier = nn.Linear(hidden_dim, num_classes)


    def forward(self, X_spatial, X_spectral):
     
        # Transform the input 
        spectral_sequence = self.spectral_sequence(X_spectral.unsqueeze(1)) #(batch_size,out_channels,seq_length)
        X_spectral = spectral_sequence.transpose(1,2)  #(batch_size, seq_length, out_channels)
        X_spatial = self.spatial_process(X_spatial) #(batch_size, )

        concatenated_features = torch.cat([Xspatial, Xspectral], dim=-1)

        spatial_attended = self.spatial_attention(Xspatial, concatenated_features)
        spectral_attended = self.spectral_attention(Xspectral, concatenated_features)

        # Normalize and combine weights
        weights = F.softmax(self.weights, dim=0)
        combined_output = weights[0] * spatial_attended + weights[1] * spectral_attended

        logits = self.classifier(combined_output.mean(dim=1))  # Assume average pooling over sequence
        return logits

   
""


class DualPathNetwork(nn.Module):
    def __init__(self, spatial_dim, spectral_dim, num_class,hidden_dim = 128):
        super(DualPathNetwork, self).__init__()
        self.out_channels = 256

        self.spatial_process = nn.Sequential(
            
            nn.Linear(spatial_dim, self.out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(num_features= self.out_channels), 
        )
        self.spectral_sequence = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features =self.out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2))
        # Position embeddings for the spectral sequence
        #self.position_embedding = nn.Parameter(torch.randn(36, self.out_channels)*0.02-0.01)

        self.norm = nn.LayerNorm(hidden_dim)
        self.query_transform = nn.Linear(self.out_channels, hidden_dim)
        self.key_transform = nn.Linear(self.out_channels, hidden_dim)
        self.value_transform = nn.Linear(self.out_channels, hidden_dim)
        self.scale = torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float))
        self.fc = nn.Linear(hidden_dim, num_class)

    def forward(self, X_spatial,X_spectral):
        # Transform the inputs
        
        spectral_sequence = self.spectral_sequence(X_spectral.unsqueeze(1)) #(batch_size,out_channels,seq_length)
        X_spectral = spectral_sequence.transpose(1,2)  #(batch_size, seq_length, out_channels)
        X_spatial = self.spatial_process(X_spatial)

        # Generate sinusoidal position embeddings
        #position = torch.arange(36).unsqueeze(1) # (:seg_length, :)
        #position = position.to(dev())
        #embedding_dim = 32
        #position_embeddings = torch.cat([torch.sin(position / (10000 ** (2 * i / embedding_dim))) if i % 2 == 0 else torch.cos(position / (10000 ** (2 * i / embedding_dim))) for i in range(embedding_dim)], dim=1)
        #position_embeddings = position_embeddings.to(dev())
        # Add fixed position embeddings to input tensor `x`
        #X_spectral += position_embeddings.unsqueeze(0)

        # Adding position embeddings
        #position_embeddings = self.position_embedding[:36, :]  # (seq_length, 32)
        #X_spectral += position_embeddings.unsqueeze(0)  # Broadcasting to add position embeddings


        query_no_norm = self.query_transform(X_spatial)  # (batch_size, hidden_dim)
        query = self.norm(query_no_norm)
        
        
        key = self.key_transform(X_spectral)        # (batch_size, seq_len, hidden_dim)
        key  = self.norm(key)
        value = self.value_transform(X_spectral)  # (batch_size, seq_len, hidden_dim)
        value = self.norm(value)

        attention_scores = torch.bmm(query.unsqueeze(1), key.transpose(-2, -1))  # (batch_size, 1, seq_len)
        attention_scores = attention_scores / self.scale  # Scaling

        # Apply softmax to get the attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, 1, seq_len)
        
        # Apply the attention weights to the values
        attended_output = torch.bmm(attention_weights, value)  # (batch_size, 1, hidden_dim)
        attended_output = attended_output.squeeze(1)

        output_with_residual = attended_output + query_no_norm # (batch_size, hidden_dim)
        # Classification
        logits = self.fc(output_with_residual)  # (batch_size, num_classes
        return logits

"""
"""
class DualPathNetwork(nn.Module):
  #accuracy 91% version with spatial attend to spectral
    def __init__(self, spatial_dim, spectral_dim, num_class,model_dim = 256, num_heads = 1):
        super().__init__()
        self.spectral_sequence = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features =32),
            nn.ReLU(),
            nn.MaxPool1d(2)

        )
        self.spectral_embed=  nn.Linear(32, model_dim)
        self.spatial_embed= nn.Linear(spatial_dim, model_dim)
        self.cross_attn = nn.MultiheadAttention(model_dim, num_heads)
        self.classifier = nn.Linear(model_dim, num_class)
            

    def forward(self, spatial_features,spectral_features):
        # Embed inputs

        spatial_features = spatial_features.unsqueeze(0)
        spectral_features = spectral_features.unsqueeze(1)
        spectral_sequence = self.spectral_sequence(spectral_features)  # (seq_length, batch, model_dim)
        spectral_sequence = spectral_sequence.permute(2,0,1)
        spectral_embed = self.spectral_embed(spectral_sequence)
        spatial_embed = self.spatial_embed(spatial_features)  # (1, batch, model_dim)

        # Cross-attention: sequence data querying non-sequence data
        attn_output, _  = self.cross_attn(spatial_embed, spectral_embed, spectral_embed)
        attn_output = attn_output.permute(1,0,2)
        attn_output = attn_output.view(attn_output.size(0), -1)
        output = self.classifier(attn_output)
        return output


"""


"""
class DualPathNetwork(nn.Module):
  # modify on July 8th 
  # naive cancantenate late fusion verison, with accuracy around 97% for uh dataset
    def __init__(self,spatial_dim,spectral_dim,num_class):
        super(DualPathNetwork, self).__init__()
        # Process spatial features
        self.spatial_processor = nn.Sequential(
            nn.Linear(spatial_dim, 256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU()
            
        )

        #self.spatial_scale = nn.Parameter(torch.ones(1))
        #self.spectral_scale = nn.Parameter(torch.ones(1))
        # Process spectral features
        self.spectral_processor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features =32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(32 * spectral_dim//4, 256),  # Assuming the length reduces to 36 after pooling
            nn.BatchNorm1d(num_features=256),
            nn.ReLU()
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512,128), 
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Linear(128, num_class)
        )


        #self.classifier = nn.Linear(512, num_class)  # Assuming 10 classes

    def forward(self, spatial_features, spectral_features):
        
        # Process each path
        spatial_out = self.spatial_processor(spatial_features)
        spectral_out = self.spectral_processor(spectral_features.unsqueeze(1))  # Add channel dimension for Conv1d
        # Combine and classify
        # Apply learnable scaling factors
        #scaled_spatial_out = self.spatial_scale * spatial_out
        #scaled_spectral_out = self.spectral_scale * spectral_out
        combined_features = torch.cat((spatial_out, spectral_out), dim=1)
        output = self.classifier(combined_features)
        return output
"""

# feature fusion # alpha beta on every dimension 
"""
class DualPathNetwork(nn.Module):
    def __init__(self, spatial_dim, spectral_dim, num_class):
        super(DualPathNetwork, self).__init__()
        # Process spatial features
        self.spatial_processor = nn.Sequential(
            nn.Linear(spatial_dim, 256),
            nn.BatchNorm1d(num_features=256),
            #nn.ReLU()
            #nn.Dropout(0.1)
        )
        
        # Process spectral features
        self.spectral_processor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(32 * (spectral_dim // 4), 256),  # Assuming the length reduces to 36 after pooling
            nn.BatchNorm1d(num_features=256),
            #nn.ReLU()
            #nn.Dropout(0.2)
        )

        # Define the class-specific weights for spatial and spectral features as learnable parameters
        self.alpha = nn.Parameter(torch.ones(256, requires_grad=True))  # Learnable weights for spatial features
        self.beta = nn.Parameter(torch.ones(256, requires_grad=True))   # Learnable weights for spectral features

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Linear(128, num_class)
        )

    def forward(self, spatial_features, spectral_features):
        # Process each path
        spatial_out = self.spatial_processor(spatial_features)
        spectral_out = self.spectral_processor(spectral_features.unsqueeze(1))  # Add channel dimension for Conv1d
        
        # Normalize alpha and beta to be between 0 and 1
        alpha_normalized = torch.sigmoid(self.alpha)
        beta_normalized = torch.sigmoid(self.beta)
        
        # Apply learned scaling factors independently
        combined_features = alpha_normalized * spatial_out + beta_normalized * spectral_out
        
        # Classification
        output = self.classifier(combined_features)
        
        return output

"""

# modify on July 11 ,easier fusion 
class DualPathNetwork(nn.Module):
    def __init__(self, spatial_dim, spectral_dim, num_class):
        super(DualPathNetwork, self).__init__()
        # Process spatial features
        self.spatial_processor = nn.Sequential(
            nn.Linear(spatial_dim, 256),
            nn.BatchNorm1d(num_features=256),
            #nn.ReLU()
            #nn.Dropout(0.1)
        )
        
        # Process spectral features
        self.spectral_processor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(64 * (spectral_dim // 4), 256),  # Assuming the length reduces to 36 after pooling
            nn.BatchNorm1d(num_features=256),
            #nn.ReLU()
            #nn.Dropout(0.2)
        )

        # Define the class-specific weights for spatial and spectral features as learnable parameters
       
        self.alpha = nn.Parameter(torch.tensor(0.7), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(0.3), requires_grad=True)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Linear(128, num_class)
        )

    def forward(self, spatial_features, spectral_features):
        # Process each path
        spatial_out = self.spatial_processor(spatial_features)
        spectral_out = self.spectral_processor(spectral_features.unsqueeze(1))  # Add channel dimension for Conv1d

        # Apply learned scaling factors independently
        combined_features = self.alpha * spatial_out + self.beta * spectral_out
        
        # Classification
        output = self.classifier(combined_features)
        
        return output


# score fusion ,even worse
"""
 # Define the DualPathNetwork
class DualPathNetwork(nn.Module):
    def __init__(self, spatial_dim, spectral_dim, num_class):
        super(DualPathNetwork, self).__init__()
        self.spatial_processor = pixel_classifier(num_class, spatial_dim)
        self.spectral_processor = Conv1D_Classfier(num_class, spectral_dim)

        # Define the class-specific weights for spatial and spectral scores as learnable parameters
        self.alpha = nn.Parameter(torch.ones(num_class, requires_grad=True))  # Learnable weights for spatial scores
        self.beta = nn.Parameter(torch.ones(num_class, requires_grad=True))   # Learnable weights for spectral scores

    def forward(self, spatial_features, spectral_features):
        # Get classification scores from each path
        spatial_scores = self.spatial_processor(spatial_features)
        spectral_scores = self.spectral_processor(spectral_features)
        
        # Normalize alpha and beta to be between 0 and 1
        alpha_normalized = torch.sigmoid(self.alpha)
        beta_normalized = torch.sigmoid(self.beta)
        
        # Apply learned scaling factors independently and combine scores
        combined_scores = alpha_normalized * spatial_scores + beta_normalized * spectral_scores
        
        return combined_scores


"""
# define a Conv1D classify spectral signal for UH dataset

"""
class Conv1D_Classfier(nn.Module):
    def __init__(self, num_classes):
        super(Conv1D_Classfier, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=16)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Second convolutional layer
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Third convolutional layer
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=64)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # Fully connected layer
        self.fc = nn.Linear(64 * 18, num_classes)  #UH dataset Adjust the sizing according to the output of the last pooling layer
      


    def forward(self, x):
        # Apply convolutional layer, batch normalization, ReLU activation, and pooling
        x = x.unsqueeze(1)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten the output for the dense layer
        x = x.view(-1, 64 * 18)  # Adjust the flatten size according to the output of the last pooling layer
        x = self.fc(x)
        return x

"""
"""
# July 11 ,make it more complex
class Conv1D_Classfier(nn.Module):
    def __init__(self, num_classes,bands_num):
        super(Conv1D_Classfier, self).__init__()
        # First convolutional layer
        self.bands_num = bands_num
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.pool1 = nn.MaxPool1d(kernel_size=2,stride = 2)

        # Second convolutional layer
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Fully connected layer
        self.fc1 = nn.Linear(64 * (self.bands_num//4), 128)  #UH dataset Adjust the sizing according to the output of the last pooling layer
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)


    def forward(self, x):
        # Apply convolutional layer, batch normalization, ReLU activation, and pooling
        x = x.unsqueeze(1)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Flatten the output for the dense layer
        x = x.view(-1, 64* (self.bands_num//4))  # Adjust the flatten size according to the output of the last pooling layer
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.fc2(x)
        return x
"""
# for Berlin dataset and Augsburg dataset
class Conv1D_Classfier(nn.Module):
    def __init__(self, num_classes,bands_num):
        super(Conv1D_Classfier, self).__init__()
        # First convolutional layer
        self.bands_num = bands_num
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=16)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Second convolutional layer
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Fully connected layer
        self.fc = nn.Linear(32 * (self.bands_num//4), num_classes)  #UH dataset Adjust the sizing according to the output of the last pooling layer
      


    def forward(self, x):
        # Apply convolutional layer, batch normalization, ReLU activation, and pooling
        x = x.unsqueeze(1)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Flatten the output for the dense layer
        x = x.view(-1, 32* (self.bands_num//4))  # Adjust the flatten size according to the output of the last pooling layer
        x = self.fc(x)
        return x

# Adopted from https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/train_interpreter.py#L68
# pixel_classfier is f_network for spatial feature classfication
class pixel_classifier(nn.Module):
    def __init__(self, numpy_class, dim):
        super(pixel_classifier, self).__init__()
        if numpy_class < 30:
            self.layers = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=32),
                nn.Linear(32, numpy_class)
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, numpy_class)
            )

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x):
        return self.layers(x)


def predict_labels(models, x_spatial, x_spectral, size):
    if isinstance(x_spatial, np.ndarray):
        x_spatial = torch.from_numpy(x_spatial)
    if isinstance(x_spectral, np.ndarray):
        x_spectral = torch.from_numpy(x_spectral)
        
    #pdb.set_trace()
    mean_seg = None
    all_seg = []
    all_entropy = []
    seg_mode_ensemble = []
    
    softmax_f = nn.Softmax(dim=1)
    with torch.no_grad():
        for MODEL_NUMBER in range(len(models)):
            #pdb.set_trace()
            #preds = models[MODEL_NUMBER](x_spatial.cuda(),x_spectral.cuda()) # if using dualnetwork model
            preds = models[MODEL_NUMBER](x_spatial.cuda())  # if using pixel classfier model
            #preds = models[MODEL_NUMBER](x_spectral.cuda()) # if using  conv1d classfier model
            entropy = Categorical(logits=preds).entropy()
            all_entropy.append(entropy)
            all_seg.append(preds)

            if mean_seg is None:
                mean_seg = softmax_f(preds)
            else:
                mean_seg += softmax_f(preds)

            img_seg = oht_to_scalar(preds)
            img_seg = img_seg.reshape(*size)
            img_seg = img_seg.cpu().detach()

            seg_mode_ensemble.append(img_seg)
        mean_seg = mean_seg / len(all_seg)

        full_entropy = Categorical(mean_seg).entropy()

        js = full_entropy - torch.mean(torch.stack(all_entropy), 0)
        top_k = js.sort()[0][- int(js.shape[0] / 10):].mean()

        img_seg_final = torch.stack(seg_mode_ensemble, dim=-1)
        img_seg_final = torch.mode(img_seg_final, 2)[0]
    return img_seg_final, top_k

def load_sorted_patches_npy(folder_path):
    # List all files in the directory
    files = [file for file in os.listdir(folder_path) if file.endswith('.npy')]

    #files = os.listdir(folder_path)
    # Sort filenames by stripping the prefix and the '.png', then converting to integer
    #sorted_filenames_img = sorted(files, key=lambda x: int(x[len("hyperspectral"):-len(".jpg")]))
    sorted_filenames_label = sorted(files, key=lambda x: int(x[len("hyperspectral"):-len(".npy")]))
    # Load images into a list
    patches_label= [np.load(os.path.join(folder_path, file)) for file in sorted_filenames_label]
    
    return patches_label


def create_inference_map(patches, width=1920, height=352, patch_size=64, overlap=32):
    """
    Construct an inference map from overlapping patches by voting.

    Parameters:
    - patches: numpy array of patches.
    - width: width of the original image.
    - height: height of the original image.
    - patch_size: size of each patch.
    - overlap: overlap between patches.

    Returns:
    - inference_map: a numpy array representing the voting result of the class labels.
    """
    stride = patch_size - overlap

    # Initialize the voting structure
    voting_map = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
   
    # Calculate patch coordinates
    patch_coords = [(i * stride, j * stride) for i in range((height - patch_size) // stride + 1)
                    for j in range((width - patch_size) // stride + 1)]

    # Fill the voting map
    for patch, (x, y) in zip(patches, patch_coords):
        for i in range(patch_size):
            for j in range(patch_size):
                pixel_class = patch[i, j]
                voting_map[x + i][y + j][pixel_class] += 1

    # Initialize the final inference map
    inference_map = np.zeros((height, width), dtype=int)

    # Determine the mode for each pixel
    for x in range(height):
        for y in range(width):
            pixel_votes = voting_map[x][y]
            if pixel_votes:
                inference_map[x, y] = max(pixel_votes, key=pixel_votes.get)

    return inference_map



def save_predictions(args, image_paths, preds):

    palette = get_palette(args['category'])
    os.makedirs(os.path.join(args['exp_dir'], 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(args['exp_dir'], 'visualizations'), exist_ok=True)

    
    prediction_path = os.path.join(args['exp_dir'],'predictions')
    print(f"save the predcitons lables to {prediction_path}")
    for i, pred in enumerate(preds):
        filename = image_paths[i].split('/')[-1].split('.')[0]
        pred = np.squeeze(pred)     
        np.save(os.path.join(prediction_path, filename + '.npy'), pred)
        if(i%100 ==0 ):
          print(f"saving {i}th file ")

    patches_npy = load_sorted_patches_npy(prediction_path)

    #inference_map = create_inference_map(patches_npy,width=1920, height=352,patch_size=64, overlap=32)
    #start_row = 352-349
    #start_col = 1920-1905

    inference_map = create_inference_map(patches_npy,width=args['img_width_adjusted'], height=args['img_height_adjusted'],patch_size=64, overlap=32)
    start_row = args['img_height_adjusted']- args['img_height_orig']
    start_col = args['img_width_adjusted'] - args['img_width_orig']

    inference_map = inference_map[start_row:, start_col:]
    np.save(os.path.join(args['exp_dir'], 'visualizations', 'inference_map.npy'),inference_map)


    test_label = np.load(args['test_label_calculate_metric'])
    ignore_label = args['ignore_label']
    valid_mask = test_label!= ignore_label
    inference_map_filted = inference_map[valid_mask]

    test_label_filted = test_label[valid_mask]-1 # don't forget to minus 1

    adjusted_inferene_map = inference_map +1
    masked_inference_map = np.where(valid_mask, adjusted_inferene_map, 0)
    
    mask = colorize_mask(masked_inference_map, palette)
    Image.fromarray(mask).save(
      os.path.join(args['exp_dir'], 'visualizations', 'inference_map.jpg')
        )
    visualization_path = os.path.join(args['exp_dir'], 'visualizations', 'inference_map.jpg')
    print(f"Save inference map to{visualization_path} ")


 
    return test_label_filted,inference_map_filted

def calculate_metric_per_class_plot_cm(args,test_label_filted, inference_map_filted):
   
    class_labels = get_class_names(args['category'])

    predictions_flat = inference_map_filted.flatten()  # prediction labels, exclude unlabeled , label =0 pixels
    test_labels_flat = test_label_filted.flatten() # true label exclude unlabeled ,label = 0 pixels

    # Create the confusion matrix
    cm = confusion_matrix(test_labels_flat, predictions_flat)

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # Saving the plot to a file
    plt.savefig(os.path.join(args['exp_dir'], 'visualizations', 'confusion_matrix.png'))  # Save as PNG file
    plt.savefig(os.path.join(args['exp_dir'], 'visualizations', 'confusion_matrix.svg'), format='svg')  # Save as SVG file for vectorized output
    plt.savefig(os.path.join(args['exp_dir'], 'visualizations', 'confusion_matrix.pdf'), format='pdf')  # Save as PDF file for documents

     # Calculate metrics
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    #TN = np.sum(cm) - (FP + FN + TP)
    #TN  = np.sum(cm) * np.ones_like(TP) - (np.sum(cm, axis=0) + np.sum(cm, axis=1) - TP)

    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    #accuracy = (TP + TN) / np.sum(cm)
    f1_scores = 2 * precision * recall / (precision + recall)
    miou = TP / (TP + FP + FN)
    overall_accuracy = np.sum(TP) / np.sum(cm)
    kappa = cohen_kappa_score(test_labels_flat, predictions_flat)

    print("Per-Class Metrics:")
    for idx, label in enumerate(class_labels):
      print(f"{class_labels[idx]} -   Recall: {recall[idx]:.4f}, Precision: {precision[idx]:.4f},F1 Score: {f1_scores[idx]:.4f}, IoU: {miou[idx]:.4f}")

    print(f"Overall accuracy: {overall_accuracy:.4f}")
    print(f"AA: {np.nanmean(recall):.4f}")
    print(f"Kappa Coefficient: {kappa:.4f}")
    print(f"Mean IoU: {np.nanmean(miou):.4f}")
    print(f"Mean F1 Score: {np.nanmean(f1_scores):.4f}")
    

    # Open the text file for writing
    with open( os.path.join(args['exp_dir'], 'visualizations', 'metrics_output.txt'), 'w') as file:
        print("Per-Class Metrics:", file=file)
        for idx, label in enumerate(class_labels):
            print(f"{class_labels[idx]} -  Recall: {recall[idx]:.4f}, Precision: {precision[idx]:.4f}, F1 Score: {f1_scores[idx]:.4f}, IoU: {miou[idx]:.4f}", file=file)
        print(f"Overall Accuracy: {overall_accuracy:.4f}", file=file)
        print(f"AA: {np.nanmean(recall):.4f}",file = file)
        print(f"Kappa Coefficient: {kappa:.4f}", file=file)
        print(f"Mean IoU: {np.nanmean(miou):.4f}", file=file)
        print(f"Mean F1 Score: {np.nanmean(f1_scores):.4f}", file=file)




#  using iou, accuracy , f1 scores to mesure model 
def calculate_metric_per_class(args,test_label, inference_map, num_classes):

    class_names = get_class_names(args['category'])
    ignore_label = args['ignore_label']
    iou_scores = np.zeros(num_classes)
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)   
    f1_scores = np.zeros(num_classes)

    mask = test_label!= ignore_label
    # Calculate accuracy only over masked (labeled) areas
    masked_pred = np.where(mask == 1, inference_map, np.nan)
    masked_gt = np.where(mask == 1, test_label, np.nan)
   

    #Calculate correct predictions
    correct_predictions = masked_gt == masked_pred
    # Calculate accuracy
    accuracy = np.sum(correct_predictions)/np.sum(mask)
    miou_list  = []
    for cls in range(1, num_classes+1):
        # Calculate intersection: True Positives (TP)
        TP = (masked_pred == cls) & (masked_gt == cls)
        TP = np.sum(TP)
        TN = np.sum((masked_pred != cls) & (masked_gt != cls))
        
        # Calculate union: TP + False Positives (FP) + False Negatives (FN)
        FP_mask = (masked_gt != cls) & (masked_pred == cls)
        FN_mask = (masked_gt == cls) & (masked_pred != cls)
        FP = np.sum(FP_mask)
        FN = np.sum(FN_mask)
       
        # Calculating Precision and Recall
        accuracy[cls] = TP/(TP+TN)
        precision[cls] = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall[cls] = TP / (TP + FN) if (TP + FN) > 0 else 0
        if (precision[cls] + recall[cls]) >0:
          f1_scores[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls])
        else:
          f1_scores[cls] = 0
        union = TP + FP + FN
        if union == 0:
            iou_scores[cls] = np.nan  # or set to zero, depending on how you want to handle this case
        else:
            iou_scores[cls] = TP / union  

        print(f" {class_names[cls]} Iou: {iou_scores[cls]:.4}, Precision: {precision[cls]:.4},F1 score : {f1_scores[cls]}, Recall: {recall[cls]}")


    return np.array(iou_scores).mean(),accuracy,recall, f1_scores




def compute_iou(args, preds, gts, print_per_class_ious=True):
    class_names = get_class_names(args['category'])

    ids = range(args['number_class'])

    unions = Counter()
    intersections = Counter()

    for pred, gt in zip(preds, gts):
        for target_num in ids:
            if target_num == args['ignore_label']: 
                continue
            preds_tmp = (pred == target_num).astype(int)
            gts_tmp = (gt == target_num).astype(int)
            unions[target_num] += (preds_tmp | gts_tmp).sum()
            intersections[target_num] += (preds_tmp & gts_tmp).sum()
    
    ious = []
    for target_num in ids:
        if target_num == args['ignore_label']: 
            continue
        iou = intersections[target_num] / (1e-8 + unions[target_num])
        ious.append(iou)
        if print_per_class_ious:
            print(f"IOU for {class_names[target_num]} {iou:.4}")
    return np.array(ious).mean()


def load_ensemble(args, device='cpu'):
    #pdb.set_trace()
    models = []
    for i in range(args['model_num']):
        model_path = os.path.join(args['exp_dir'], f'model_{i}.pth')
        state_dict = torch.load(model_path)['model_state_dict']

        #model = DualPathNetwork(spatial_dim = args['dim'][-1],spectral_dim = args['bands_num'], num_class = args['number_class'])  # using naive feature conca
        #model = DualPathNetwork(num_class =args['number_class']) # using cross attention 
        #model = DualPathNetwork_cross(num_class =args['number_class']) # using cross attention ,for berlin dataset
        model = pixel_classifier(numpy_class= args['number_class'], dim= args['dim'][-1]) # if using pixel classfier
        #model = Conv1D_Classfier(num_classes= args['number_class'],bands_num = args['bands_num']) # if using Conv1d classfier
        
        model.load_state_dict(state_dict)
        model = model.to(device)
        models.append(model.eval())
    return models
