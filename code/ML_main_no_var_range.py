import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, accuracy_score,precision_score,f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
def get_acc(y_pred, y_true):
    t=0
    for i in range(len(y_pred)):
        if y_pred[i]==y_true[i]:
            t+=1
    acc=t/len(y_pred)
    return acc

def get_ml_input(data,level=2):
    x=data[['left_temp',
            'right_temp', 
            'Disorder', 'sex', 'age', 'from_19_min', 
            'log_left_ENMO_0','left_anglex_0', 'left_angley_0', 'left_anglez_0', 
            'log_right_ENMO_0','right_anglex_0', 'right_angley_0', 'right_anglez_0',
            'log_left_ENMO_1', 'left_anglex_1', 'left_angley_1', 'left_anglez_1',
            'log_right_ENMO_1', 'right_anglex_1', 'right_angley_1', 'right_anglez_1',
            'log_left_ENMO_2', 'left_anglex_2', 'left_angley_2','left_anglez_2', 
            'log_right_ENMO_2', 'right_anglex_2', 'right_angley_2','right_anglez_2', 
            'log_left_ENMO_3', 'left_anglex_3','left_angley_3', 'left_anglez_3', 
            'log_right_ENMO_3', 'right_anglex_3','right_angley_3', 'right_anglez_3',  
            'log_left_ENMO_4','left_anglex_4', 'left_angley_4', 'left_anglez_4', 
            'log_right_ENMO_4','right_anglex_4', 'right_angley_4', 'right_anglez_4', 
            'log_left_ENMO_5', 'left_anglex_5', 'left_angley_5', 'left_anglez_5',
            'log_right_ENMO_5', 'right_anglex_5', 'right_angley_5', 'right_anglez_5']].values
    if level==2:
        y=data["b_y"].values
    elif level==3:
        y=data["t_y"].values
    return x,y

data=pd.read_csv("./data/data_concated_log.csv",index_col=0,header=0)
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
leave_one_tr_acc=[]
leave_one_val_acc=[]
leave_one_precision=[]
leave_one_recall=[]
leave_one_f=[]

for i in range(len(nums)):
    print("Running on",i)
    val_nums=[nums.pop(i)]
    tr_nums=nums.copy()
    nums=[1,2,14,17,21,23,27,28,29,31,32,34,35,38,39,42,45,48,49,50,51,52,53,56,57,59,60]
    data_tr=data[data["participant_id"].isin(tr_nums)]
    data_val=data[data["participant_id"].isin(val_nums)]
    x_val, y_val=get_ml_input(data_val,level=3)
    x_tr, y_tr=get_ml_input(data_tr,level=3)
    """
    #PCA
    pca = PCA(n_components=3)
    pca.fit(x_tr)
    p_x_tr = pca.transform(x_tr)
    p_x_val = pca.transform(x_val)
    """
    # Select model
    #clf = Perceptron(verbose=1)
    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(64,32), random_state=1)
    #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
    #clf = SVC(gamma='auto',kernel="linear")
    #clf = SVC(gamma='auto',kernel="rbf")
    #clf = SVC(gamma='auto',kernel="sigmoid")
    #clf=SVC(gamma='auto',kernel="rbf",class_weight={0:0.1, 1:0.9})
    #clf = MultinomialNB()
    #clf = GaussianNB()
    clf=LogisticRegression()
    #clf = LogisticRegression(class_weight="balanced")
    #clf=ExtraTreeClassifier()
    #clf=RandomForestClassifier()
    
    clf.fit(x_tr, y_tr)
    pred_val=clf.predict(x_val)   
    pred_tr=clf.predict(x_tr)
    leave_one_tr_acc.append(get_acc(pred_tr, y_tr))
    leave_one_val_acc.append(get_acc(pred_val, y_val))

f=open("./3_level_results.txt","a+")
f.write("LR_no_var_no_range_all_samples\n")
f.write("average tr acc:"+str(np.mean(leave_one_tr_acc))+"\n")
f.write("average val acc:"+str(np.mean(leave_one_val_acc))+"\n")
f.write("Tr acc:"+",".join(str(x) for x in leave_one_tr_acc)+"\n")
f.write("Val acc:"+",".join(str(x) for x in leave_one_val_acc)+"\n\n")
f.close()