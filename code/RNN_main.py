from RNN import *
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score, recall_score, f1_score

def get_acc(y_pred, y_true):
    t=0
    for i in range(len(y_pred)):
        if y_pred[i]==y_true[i]:
            t+=1
    acc=t/len(y_pred)
    return acc

########### Read in data ################
data=pd.read_csv("./data/data_forRNN_log_more.csv",index_col=0,header=0)
#data['left_ENMO-1']=1/(data["left_ENMO"]+0.000000001)
#data['right_ENMO-1']=1/(data["right_ENMO"]+0.000000001)
#Normalize data
data=normalize(data)
#Add binary and tri y
# 0 for wake, 1 for sleep
data["b_y"]=(data["y"]!=0)*1
# 0 for wake, 1 for REM, 2 for non-REM123
t_y=[]
for i in data["y"]:
    if i==0: t_y.append(0)
    elif i==4: t_y.append(1)
    else: t_y.append(2)

data["t_y"]=t_y

#---------------------------------------------------------------------------------
back=[0,5,10,20,40]
front=0
for b in back:
    leave1_val_acc=[]
    leave1_tr_acc=[]
    leave1_val_precision=[]
    leave1_val_recall=[]
    leave1_val_f1=[]
    leave1_val_auc=[]
    nums=[1,2,14,17,21,23,27,28,29,31,32,34,35,38,39,42,45,48,49,50,51,52,53,56,57,59,60]
    #leave one validation
    for i in range(len(nums)):
        val_nums=[nums.pop(i)]
        tr_nums=nums.copy()
        nums=[1,2,14,17,21,23,27,28,29,31,32,34,35,38,39,42,45,48,49,50,51,52,53,56,57,59,60]
        data_tr=data[data["participant_id"].isin(tr_nums)]
        data_val=data[data["participant_id"].isin(val_nums)]
        X_train, y_train = get_RNN_input(data_tr,nums=tr_nums, back=b, front=front,level=3)
        X_val, y_val = get_RNN_input(data_val,nums=val_nums , back=b, front=front,level=3)
    
        #Prepare data loader
        batch_size=32
        train_dataset=Dataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
        val_dataset=Dataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)
    
        # Run Baseline RNN
        model=RNNClassify(input_size=22, hidden_size=128, hidden_size2=64,output_size=3,batch_size=batch_size).to(device)
        #model=CharGRULSTM(input_size=22, hidden_size=256, output_size=3, seq_len=6).to(device)
    
        #criterion=nn.CrossEntropyLoss().to(device)
        criterion=nn.NLLLoss().to(device)
        learning_rate=0.001
        #optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        optimizer= optim.Adagrad(model.parameters(),lr=learning_rate)
        num_epochs=10
        model_trained,acc_train,loss_train, acc_val=train(model=model, train_loader=train_loader, validation_loader=val_loader, loss_function=criterion, optimizer=optimizer,num_epochs=num_epochs)
        X_val=val_loader.dataset.x_data.to(device)
        pred_val, prob_val=predict(model_trained, X_val)
        pred_val=[i[0] for i in pred_val.tolist()]

        leave1_tr_acc.append(acc_train[-1])  
        leave1_val_acc.append(get_acc(pred_val, y_val.tolist()))
        leave1_val_precision.append(precision_score(y_val.tolist(),pred_val,average="weighted"))
        leave1_val_recall.append(recall_score(y_val.tolist(),pred_val,average="weighted"))
        leave1_val_f1.append(f1_score(y_val.tolist(),pred_val,average="weighted"))

        if len(set(y_val))==prob_val.shape[1]:
            #One hot
            dummy=np.concatenate((((y_val==0)*1).reshape(-1,1),((y_val==1)*1).reshape(-1,1),((y_val==2)*1).reshape(-1,1)),axis=1)
            leave1_val_auc.append(roc_auc_score(dummy,prob_val.cpu().numpy()))
        else:
            print("Sample"+str(nums[i])+"do not have AUC!")


    f=open("./3_level_RNN_results.txt","a+")
    f.write("RNN,b="+str(b)+",f=0,128_64\n")
    f.write("average tr acc:"+str(np.mean(leave1_tr_acc))+"\n")
    f.write("average val acc:"+str(np.mean(leave1_val_acc))+"\n")
    f.write("average val precision:"+str(np.mean(leave1_val_precision))+"\n")
    f.write("average val recall:"+str(np.mean(leave1_val_recall))+"\n")
    f.write("average val f1:"+str(np.mean(leave1_val_f1))+"\n")
    f.write("average val auc:"+str(np.mean(leave1_val_auc))+"\n\n")
    f.close()

    f=open("./3_level_6_metrics_RNN_details.txt","a+")
    f.write("RNN,b="+str(b)+",f=0,128_64\n")
    f.write("Tr acc:"+",".join(str(x) for x in leave1_tr_acc)+"\n")
    f.write("Val acc:"+",".join(str(x) for x in leave1_val_acc)+"\n\n")
    f.write("Val precision:"+",".join(str(x) for x in leave1_val_precision)+"\n\n")
    f.write("Val recall:"+",".join(str(x) for x in leave1_val_recall)+"\n\n")
    f.write("Val f1:"+",".join(str(x) for x in leave1_val_f1)+"\n\n")
    f.write("Val auc:"+",".join(str(x) for x in leave1_val_auc)+"\n\n")
    f.close()
#---------------------------------------------------------------------------------


