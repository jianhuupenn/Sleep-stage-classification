from LSTM import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
nums=[1,2,14,17,21,23,27,28,29,31,32,34,35,38,39,42,45,48,49,50,51,52,53,56,57,59,60]
b=50
#back=[40,50,60,70,80]
front=[0]
for f in front:
    leave1_val_acc=[]
    leave1_tr_acc=[]
    leave1_val_precision=[]
    leave1_val_recall=[]
    leave1_val_f1=[]
    leave1_val_auc=[]
    nums=[1,2,14,17,21,23,27,28,29,31,32,34,35,38,39,42,45,48,49,50,51,52,53,56,57,59,60]
    #nums=[1,2,14]
    #leave one validation
    for i in range(len(nums)):
        print("Doing sample ",str(i))
        val_nums=[nums.pop(i)]
        tr_nums=nums
        nums=[1,2,14,17,21,23,27,28,29,31,32,34,35,38,39,42,45,48,49,50,51,52,53,56,57,59,60]
        #nums=[1,2,14]
        data_tr=data[data["participant_id"].isin(tr_nums)]
        data_val=data[data["participant_id"].isin(val_nums)]
        X_train, y_train = get_LSTM_input(data_tr,nums=tr_nums,back=b, front=f,level=3)
        X_val, y_val = get_LSTM_input(data_val,nums=val_nums,back=b, front=f,level=3)
        #Prepare data loader
        batch_size=16
        train_dataset=Dataset(X_train, y_train)
        
        #Input size-------
        input_size=X_train.shape[2]
        del X_train
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
        val_dataset=Dataset(X_val, y_val)
        del X_val
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)
    
        # Run Model
        #model=RNNClassify(input_size=28, hidden_size=64, output_size=2).to(device)
        #model=CharGRULSTM(input_size=22, hidden_size=256, output_size=3, seq_len=6).to(device)
        model=CharGRULSTM(input_size=input_size, hidden_size=128,hidden_size2=64, output_size=3, seq_len=6).to(device)
        #criterion=nn.CrossEntropyLoss().to(device)
        criterion=nn.NLLLoss().to(device)
        learning_rate=0.001
        #optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        optimizer= optim.Adagrad(model.parameters(),lr=learning_rate)
        num_epochs=10
        model_trained,acc_train,loss_train, acc_val=train(model=model, train_loader=train_loader, validation_loader=val_loader, loss_function=criterion, optimizer=optimizer,num_epochs=num_epochs)
        
        #X_val=val_loader.dataset.x_data.to(device)
        pred_val, prob_val =predict(model_trained, val_loader)
        pred_val=[i[0] for i in pred_val.tolist()]
        leave1_tr_acc.append(acc_train[-1])  
        leave1_val_acc.append(calculateAccuracy(pred_val, y_val.tolist()))
        leave1_val_precision.append(precision_score(y_val.tolist(),pred_val,average="weighted"))
        leave1_val_recall.append(recall_score(y_val.tolist(),pred_val,average="weighted"))
        leave1_val_f1.append(f1_score(y_val.tolist(),pred_val,average="weighted"))

        #Sample 35 do not have AUC!
        if len(set(y_val))==prob_val.shape[1]:
            #One hot
            dummy=np.concatenate((((y_val==0)*1).reshape(-1,1),((y_val==1)*1).reshape(-1,1),((y_val==2)*1).reshape(-1,1)),axis=1)
            leave1_val_auc.append(roc_auc_score(dummy,prob_val.cpu().numpy()))
        else:
            print("Sample"+str(nums[i])+"do not have AUC!")


        #Write each patient's results
        pred_val=[str(i) for i in pred_val]
        file=open("./LSTM_128_64_b="+str(b)+"_f="+str(f)+"_patient_"+str(i)+".txt","w")
        file.write(",".join(pred_val))
        file.close()

    """
    print(leave1_val_acc)
    print(leave1_tr_acc)
    print(np.mean(leave1_val_acc))
    
    print("Tr acc ", round(np.mean(leave1_tr_acc),5))
    print("Val acc ", round(np.mean(leave1_val_acc),5))
    """
    f=open("./3_level_6_metrics.txt","a+")
    f.write("LSTM,more,b="+str(b)+",f="+str(front)+" 128_64\n")
    f.write("average tr acc:"+str(np.mean(leave1_tr_acc))+"\n")
    f.write("average val acc:"+str(np.mean(leave1_val_acc))+"\n")
    f.write("average val precision:"+str(np.mean(leave1_val_precision))+"\n")
    f.write("average val recall:"+str(np.mean(leave1_val_recall))+"\n")
    f.write("average val f1:"+str(np.mean(leave1_val_f1))+"\n")
    f.write("average val auc:"+str(np.mean(leave1_val_auc))+"\n\n")
    #f.write("Tr acc:"+",".join(str(x) for x in leave1_tr_acc)+"\n")
    #f.write("Val acc:"+",".join(str(x) for x in leave1_val_acc)+"\n\n")
    f.close()

    f=open("./3_level_6_metrics_details.txt","a+")
    f.write("LSTM,less,b="+str(b)+",f="+str(front)+",128_64\n")
    f.write("Tr acc:"+",".join(str(x) for x in leave1_tr_acc)+"\n")
    f.write("Val acc:"+",".join(str(x) for x in leave1_val_acc)+"\n\n")
    f.write("Val precision:"+",".join(str(x) for x in leave1_val_precision)+"\n\n")
    f.write("Val recall:"+",".join(str(x) for x in leave1_val_recall)+"\n\n")
    f.write("Val f1:"+",".join(str(x) for x in leave1_val_f1)+"\n\n")
    f.write("Val auc:"+",".join(str(x) for x in leave1_val_auc)+"\n\n")
    f.close()

