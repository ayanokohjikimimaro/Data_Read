# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 08:32:04 2018

@author: Satoshi KIMURA
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import glob
from matplotlib.colors import ListedColormap
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import pickle
from IPython.display import Image
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn import svm
import scipy as sp
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

        
dirnow = os.path.abspath(".")
datadir = "speech_commands_v0.01"
#filename_list = "file_name_list_0.txt"

data_num = 100
print("data_num: %d"%(data_num))

LABEL = ["on", "off", "up", "down", "yes", "no", "left", "right", "go", "stop"]
print("Clasification: %s"%str(LABEL))

feat_method = ["ave", "min", "max", "var", "ske", "kur"]
print("feat_method: %s"%str(feat_method))

n = 13#the top 13 Mel-frequency cepstral coefficients (MFCCs)
print("MFCC order: %d"%n)

f_Num = 20  # ランダムフォレス
print("Feature Num: %d"%f_Num)

n_comp = 3  # PCA
print("PCA Num: %d"%n_comp)


feature_ext = 1

start = time.time()
if feature_ext == 1:
    
    M_h = np.zeros(n*3*len(feat_method))
    M_list = np.zeros([data_num*len(LABEL), n*3*len(feat_method)])# 3(mfcc,Δ，ΔΔ)
    L_list = np.zeros([data_num*len(LABEL), 1])
    
    for iii in range(len(LABEL)):
        WFL = glob.glob( dirnow +"/"+ datadir +"/"+ LABEL[iii] +"/*.wav" )
        WF_ex = WFL[0:data_num]
        
        for jjj in range(len(WF_ex)):
            y, sr = librosa.load(WF_ex[jjj], sr=16000)  #Fs=16khz 
            
            # MFCC 
            # http://nbviewer.jupyter.org/github/librosa/librosa/blob/master/examples/LibROSA%20demo.ipynb
            # Next, we'll extract the top 13 Mel-frequency cepstral coefficients (MFCCs)
            mfcc        = librosa.feature.mfcc(y=y, n_mfcc=n)
            # Let's pad on the first and second deltas while we're at it
            delta_mfcc  = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
                   
            # For future use, we'll stack these together into one matrix
            M = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
            
            for kkk in range(len(feat_method)):
                if kkk == 0:  # ave
                    M_stati = np.mean(M, 1).reshape(1, len(M))
                elif kkk == 1:  # max
                    M_stati = np.max(M, 1).reshape(1, len(M))
                elif kkk == 2:  # min
                    M_stati = np.min(M, 1).reshape(1, len(M))
                elif kkk == 3:  # var
                    M_stati = np.var(M, 1).reshape(1, len(M))
                elif kkk == 4:  # ske
                    M_stati = sp.stats.skew(M, 1).reshape(1, len(M))
                elif kkk == 5:  # kur
                    M_stati = sp.stats.kurtosis(M, 1).reshape(1, len(M))
                elif kkk == 6:  # kur
                    M_stati = ( np.max(M, 1)-np.min(M, 1) ).reshape(1, len(M))
                elif kkk == 7:  # kur
                    M_stati = ( np.max(M, 1) / np.min(M, 1) ).reshape(1, len(M))
                    
                M_h[n*3*kkk : n*3*kkk+n*3] = M_stati
        
            M_list[jjj + len(WF_ex)*iii, :] = M_h
            L_list[jjj + len(WF_ex)*iii, :] = int(iii)

    # data_set = np.hstack([L_list, M_list])
    # np.savetxt('data_set.csv',data_set,delimiter=',')
    
    # =============================================================================
    # Random Forest
    # =============================================================================
    scr = StandardScaler()
    scr.fit(M_list)
    X_std = scr.transform(M_list)

    forest = RandomForestRegressor(n_estimators=1000,
                                  random_state=0,
                                   n_jobs = -1)

    y = L_list.ravel()
    forest.fit(X_std, y)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    
    # =============================================================================
    # Data split
    # =============================================================================
    feature_sel=[]
    for lll in range(f_Num):
        feature_sel.append(indices[lll])
    
    #feature_sel = [d1, d2]
    print("feature selected:%s"%(str(feature_sel)))
    X = M_list[:, feature_sel]#importance TOP5
    
    test_prop = 0.3
    random_seed = 1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_prop, random_state=random_seed)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    
for max_depth in [5]:    
    #print("n_components = %d"%(n_comp))
    print("max_depth = %d"%(max_depth))    
    
    # Decisiontree　Classifier
    tree = []
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=1)
    tree.fit(X_train_std, y_train)
    
    y_test_pred = tree.predict(X_test_std)
    accuracy = metrics.accuracy_score(y_test, y_test_pred)
    print("accuracy by tree : {}".format(accuracy))
    
    # 混合行列
    predicted = tree.predict(X_train_std)
    cm = confusion_matrix(y_train, predicted)
    cm_nomalized_trn = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
    
    predicted = tree.predict(X_test_std)
    cm = confusion_matrix(y_test, predicted)
    cm_nomalized_tst = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    df_confmat_trn = pd.DataFrame(cm_nomalized_trn)
    df_confmat_trn.columns = LABEL
    df_confmat_trn.index = LABEL
    sns.heatmap(df_confmat_trn, annot=True, fmt='1.1f')
    plt.title('train data')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    plt.subplot(1,2,2)
    df_confmat_tst = pd.DataFrame(cm_nomalized_tst)
    df_confmat_tst.columns = LABEL
    df_confmat_tst.index = LABEL
    sns.heatmap(df_confmat_tst, annot=True, fmt='1.1f')
    plt.title('test data')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    



# 交差検証 Dicision Tree
#tree = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=1)
scores = cross_val_score(tree, X, y, cv=10)
print('Cross-Validation scores: {}'.format(scores))
print('Average score: {}'.format(np.mean(scores)))

elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")