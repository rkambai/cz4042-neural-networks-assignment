### THIS FILE CONTAINS COMMON FUNCTIONS, CLASSSES

import tqdm
import time
import random 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from scipy.io import wavfile as wav

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix



def split_dataset(df, columns_to_drop, test_size, random_state):
    label_encoder = preprocessing.LabelEncoder()

    df['label'] = label_encoder.fit_transform(df['label'])

    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

    df_train2 = df_train.drop(columns_to_drop,axis=1)
    y_train2 = df_train['label'].to_numpy()

    df_test2 = df_test.drop(columns_to_drop,axis=1)
    y_test2 = df_test['label'].to_numpy() 

    return df_train2, y_train2, df_test2, y_test2

def preprocess_dataset(df_train, df_test):

    standard_scaler = preprocessing.StandardScaler()
    df_train_scaled = standard_scaler.fit_transform(df_train)

    df_test_scaled = standard_scaler.transform(df_test)

    return df_train_scaled, df_test_scaled

def set_seed(seed = 0):
    '''
    set random seed
    '''
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def intialise_loaders(X_train_scaled, y_train, X_test_scaled, y_test, batch_size = 256):
    # YOUR CODE HERE
    train_ds = CustomDataset(X_train_scaled, y_train)
    train_dataloader = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
    test_ds = CustomDataset(X_test_scaled, y_test)
    test_dataloader = DataLoader(test_ds, batch_size = batch_size, shuffle = True)

    return train_dataloader, test_dataloader


# early stopping obtained from tutorial
class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class MLP(nn.Module):

    def __init__(self, no_features, no_hidden, no_labels, no_hidden_neurons = (128, 128, 128)):
        super().__init__()

        assert len(no_hidden_neurons)==no_hidden, "len of number of hidden neurons must match number of hidden layers"

        input_layer = nn.Linear(no_features, no_hidden_neurons[0])
        input_activation = nn.ReLU()
        input_dropout = nn.Dropout(0.2)
        output = nn.Linear(no_hidden_neurons[-1], no_labels)
        output_activation = nn.Sigmoid()

        self.mlp_stack = nn.Sequential()
        self.mlp_stack.append(input_layer)
        self.mlp_stack.append(input_activation)
        self.mlp_stack.append(input_dropout)
        
        for i in range(no_hidden-1):
            layer = nn.Linear(no_hidden_neurons[i], no_hidden_neurons[i+1])
            activation = nn.ReLU()
            dropout = nn.Dropout(0.2)
            self.mlp_stack.append(layer)
            self.mlp_stack.append(activation)
            self.mlp_stack.append(dropout)
        self.mlp_stack.append(output)
        self.mlp_stack.append(output_activation)

    def forward(self, x):
        return self.mlp_stack(x)
    

class CustomDataset(Dataset):
    # YOUR CODE HERE
    def __init__(self, X, y):
        """
        X: input features dataframe
        y: labels dataframe
        """
        self.features = torch.tensor(X, dtype=torch.float)
        self.labels = torch.tensor(y, dtype=torch.float)
        return

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    