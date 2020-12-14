########### Define RNN ################
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import codecs
import math
import random
import string
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score


#RNN
class RNNClassify(nn.Module):
    def __init__(self, input_size, hidden_size,hidden_size2,output_size,batch_size,rnn_type = "simple"):
        super(RNNClassify, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.rnntype = rnn_type
        self.batch_size = batch_size
        #Define Layers
        self.c2h2 = nn.Linear(input_size + hidden_size, hidden_size2)
        self.i2o = nn.Linear(hidden_size2, output_size)
        self.i2h = nn.Linear(hidden_size2, hidden_size)
        self.softmax = F.log_softmax
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, input, hidden=None):
        #print("Input", input.shape)
        #print("Hidden", hidden.shape)
        combined = torch.cat((input, hidden), 1)
        hidden2 = self.c2h2(combined)
        #hidden2 = self.dropout(hidden2)
        hidden = self.i2h(hidden2)
        output = self.i2o(hidden2)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)


########### Functions and packages for RNN ################

def normalize(data):
    names1=['left_temp',
       'right_temp',  'from_19_min', 
       'log_left_ENMO', 'log_right_ENMO',
       'log_left_ENMO_var', 'log_right_ENMO_var',
       'log_left_anglex_var', 'log_right_anglex_var', 
       'log_left_angley_var', 'log_right_angley_var',
       'log_left_anglez_var', 'log_right_anglez_var', 
       'left_anglex', 'left_angley', 'left_anglez',
       'right_anglex', 'right_angley', 'right_anglez']
    names2=['participant_id','Disorder', 'sex', 'age','p_left','p_right', 'p_prone', 'p_upright','y']
    nums=[1,2,14,17,21,23,27,28,29,31,32,34,35,38,39,42,45,48,49,50,51,52,53,56,57,59,60]
    data_normalized=pd.DataFrame()
    others=pd.DataFrame()
    for num in nums:
        tmp=data[names1][data["participant_id"]==num]
        tmp=(tmp-tmp.mean())/tmp.std()
        data_normalized=pd.concat([data_normalized, tmp],axis=0,ignore_index=True)
        others_tmp=data[names2][data["participant_id"]==num]
        others=pd.concat([others, others_tmp],axis=0,ignore_index=True)
    data_normalized=pd.concat([others,data_normalized],axis=1)
    return data_normalized

# Get expanded input
def get_RNN_input(data, nums, back=0, front=0, level=2):
    names=['left_temp','right_temp', 'Disorder', 'sex', 
        'age', 'from_19_min', 'log_left_ENMO_var', 
        'log_right_ENMO_var','log_left_anglex_var', 
        'log_right_anglex_var', 'log_left_angley_var', 
        'log_right_angley_var','log_left_anglez_var', 
        'log_right_anglez_var', 'left_anglex', 
        'left_angley', 'left_anglez','right_anglex', 
        'right_angley', 'right_anglez','log_left_ENMO', 'log_right_ENMO']
    X_list=[]
    y_list=[]
    for num in nums:
        #X
        sub=data[data["participant_id"]==num]
        X=sub[names].values.astype(float)
        X=np.reshape(X,(-1,6,len(names)))
        n=int(X.shape[0])
        # get original X
        exp_X=X[back:(n-front),:,:]

        # Add back
        for i in range(1,back+1):
            back_tmp=X[back-i:n-front-i,:,:]
            exp_X=np.concatenate((back_tmp,exp_X),axis=1)
        #print(ret_X.shape)

        # Add Front
        for i in range(1,front+1):
            front_tmp=X[back+i:n-front+i,:,:]
            exp_X=np.concatenate((exp_X, front_tmp),axis=1)
        #print(ret_X.shape)
        X_list.append(exp_X)
        
        # Y
        if level==2:
            y=sub[["b_y"]].values.astype(float)
        elif level==3:
            y=sub[["t_y"]].values.astype(float)
        y=y[0::6]
        y=np.array([i[0] for i in y])
        y=y[back:n-front]
        y_list.append(y)
    
    ret_X=np.concatenate(X_list,axis=0)
    ret_y=np.concatenate(y_list,axis=0)
    return ret_X,ret_y

class Dataset(Dataset):
    def __init__(self, X, y):
        self.len = len(X)           
        if torch.cuda.is_available():
            self.x_data = torch.from_numpy(X).float().cuda()
            self.y_data = torch.from_numpy(y).long().cuda()
        else:
            self.x_data = torch.from_numpy(X).float()
            self.y_data = torch.from_numpy(y).long()
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

def calculateAccuracy(y_pred, y):
    #y_pred and y are lists
    tmp=[1 if y_pred[i]==y[i] else 0 for i in range(len(y))]
    return sum(tmp)/len(y)

def predict(model, X):
    #X in shape n*6*24, all on device
    preds=torch.torch.FloatTensor([]).to(device)
    hidden = model.init_hidden(X.shape[0]).to(device)
    for j in range(X.shape[1]):
        pred_y, hidden = model(X[:,j,:], hidden)

    preds=torch.cat((preds,pred_y),0)
    _, top_i =  preds.data.topk(1)
    return top_i, preds.data

def train(model, train_loader, validation_loader, loss_function, optimizer,num_epochs):
    total_step = len(train_loader)
    acc_train=[]
    loss_train=[]
    acc_val=[]
    #X_train=train_loader.dataset.x_data.to(device)
    #y_train=train_loader.dataset.y_data.to(device)
    # This took too much RAM!
    X_val=validation_loader.dataset.x_data.to(device)
    y_val=validation_loader.dataset.y_data.to(device)
    #Initial hidden
    for epoch in range(num_epochs):
        y_pred_train=torch.torch.FloatTensor([]).to(device)
        y_train=torch.torch.LongTensor([]).to(device)
        for i, (records, labels) in enumerate(train_loader): 
            optimizer.zero_grad()
            hidden = model.init_hidden(batch_size=records.shape[0]).to(device)
            # Move tensors to the configured device
            records = records.to(device)
            labels = labels.to(device)
            # Forward pass with data in 30s
            n=records.shape[1]
            for j in range(n):
                pred_y, hidden = model(records[:,j,:], hidden)
            loss = loss_function(pred_y, labels)
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            y_pred_train=torch.cat((y_pred_train,pred_y),0)
            y_train=torch.cat((y_train,labels),0)
         
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        #y_pred_train=model(X_train) #Cost too much RAM!
        _, top_i_train =  y_pred_train.data.topk(1)
        acc_train_tmp=calculateAccuracy(top_i_train[:,0].tolist(), y_train.tolist())
        acc_train.append(acc_train_tmp)
        
        loss_epoch=loss_function(y_pred_train, y_train).item()
        loss_train.append(loss_epoch)
        
        top_i_val, _=predict(model, X_val)
        acc_val_tmp=calculateAccuracy(top_i_val[:,0].tolist(), y_val.tolist())
        acc_val.append(acc_val_tmp)
        
    return model, np.array(acc_train),np.array(loss_train), np.array(acc_val)
