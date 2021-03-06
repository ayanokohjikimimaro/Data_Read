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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from scipy import signal
from sklearn.linear_model import LogisticRegression
import datetime

import warnings
warnings.filterwarnings('ignore')

class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.3, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, 
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, 
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

def calculate_melsp(x, n_fft=64, hop_length=32, n_mels=128):#1024,128
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=n_mels)
    return melsp

dirnow = os.path.abspath(".")
datadir = "speech_commands_v0.01"

filename_for_res = "AI_woh_1902.csv"
fp = open(filename_for_res,"a")
fp.write(str(datetime.datetime.now())+'\n')
fp.write("feature selected num, mooving_ave, num of FFT, shift, mfcc_n , 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000\n")
fp.close()




LABEL = ["on", "off", "up", "down", "yes", "no", "left", "right", "go", "stop"]
print("Clasification: %s"%str(LABEL))

data_num = 1000
print("data_num: %d"%(data_num*len(LABEL)))

feat_method = ["ave", "min", "max", "var", "ske", "kur"]
print("feat_method: %s"%str(feat_method))

n = 13#the top 13 Mel-frequency cepstral coefficients (MFCCs)
#print("MFCC order: %d"%n)

f_Num = 150  # ランダムフォレス
#print("Feature Num: %d"%f_Num)

n_comp = 3  # PCA
#print("PCA Num: %d"%n_comp)

sampelrate = 16000
new_sample_rate = int(sampelrate/2)

###################
feature_ext = 1
###################



#k_n_ = 9
#num_ = np.arange(1,12,1)
#num = [4] # mooving_average
#acc_knn_lis_puls_train = np.zeros([len(num_), k_n_-1])
#acc_knn_lis_puls_test = np.zeros([len(num_), k_n_-1])

## mfcc sweep 
#mfcc_n = [6, 13, 20]#np.arange(8,28,2)
##acc_knn_lis_puls_train = np.zeros([len(mfcc_n), k_n_])
##acc_knn_lis_puls_test = np.zeros([len(mfcc_n), k_n_])



start = time.time()
if feature_ext == 1:
    #mmm = 0
    for num in [4] : # mooving_average
        for n_fft_ in[256, 272]:#230, 256, 272
            for shift in [0.325, 0.335]:#0.3125, 0.325 ,0.325
                for n in [13, 15, 17] :#13, 15, 17
                    #print("num: %d"%(num)) # mooving_average
                    #print("mooving_ave: %d"%(num))
                    #print("num of FFT: %d"%(n_fft_))
                    #print("shift: %f"%(shift))
                    #print("mfcc_n: %d"%(n))
                
                    M_h = np.zeros(n*3*len(feat_method))
                    M_list = np.zeros([data_num*len(LABEL), n*3*len(feat_method)])# 3(mfcc,Δ，ΔΔ)
                    L_list = np.zeros([data_num*len(LABEL), 1])
                    
                    for iii in range(len(LABEL)):
                        WFL = glob.glob( dirnow +"/"+ datadir +"/"+ LABEL[iii] +"/*.wav" )
                        ##WF_ex = WFL[0:data_num]
                        
                        ##for jjj in range(len(WF_ex)):
                        jjj = 0
                        jjj_= 0
                        
                        while jjj_ < data_num:
                            y, sr = librosa.load(WFL[jjj], sr=16000)  #Fs=16khz 
                        
                            # moving average
                            b=np.ones(num)/num
                            y = np.convolve(y, b, mode='same')
                        
                            if (y.shape[0] != sampelrate) or np.std(y)*6 < 0.01:
                                #print(jjj)
                                jjj += 1
                                continue
                            
                            y = signal.resample(y, int(new_sample_rate/sampelrate * y.shape[0]))                
                        
                            # MFCC 
                            # http://nbviewer.jupyter.org/github/librosa/librosa/blob/master/examples/LibROSA%20demo.ipynb
                            # Next, we'll extract the top 13 Mel-frequency cepstral coefficients (MFCCs)
                            #mfcc        = librosa.feature.mfcc(y=y, n_mfcc=n)
                            
                            # def calculate_melsp(x, n_fft=64, hop_length=32, n_mels=128):#1024,128
                            mfcc = librosa.feature.mfcc(S=calculate_melsp( y, n_fft=n_fft_, hop_length=int(n_fft_ * shift), n_mels=128 ) , 
                                                                        n_mfcc=n )
                            
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
                                elif kkk == 6:  
                                    M_stati = ( np.max(M, 1)-np.min(M, 1) ).reshape(1, len(M))
                                elif kkk == 7:  
                                    M_stati = ( np.max(M, 1) / np.min(M, 1) ).reshape(1, len(M))
                                    
                                M_h[n*3*kkk : n*3*kkk+n*3] = M_stati
                        
                            M_list[jjj_ + data_num*iii, :] = M_h
                            L_list[jjj_ + data_num*iii, :] = int(iii)
                            
                            jjj += 1
                            jjj_+= 1
                
                    # data_set = np.hstack([L_list, M_list])
                    # np.savetxt('data_set.csv',data_set,delimiter=',')
                    
                    # =============================================================================
                    # Random Forest
                    # =============================================================================
                    scr = StandardScaler()
                    scr.fit(M_list)
                    X_std = scr.transform(M_list)
                
#                    forest = RandomForestRegressor(n_estimators=1000,
#                                                  random_state=0,
#                                                   n_jobs = -1)
                    forest = RandomForestClassifier(n_estimators=1000,
                                                  random_state=0,
                                                   n_jobs = -1)
                
                    y = L_list.ravel()
                    forest.fit(X_std, y)
                    importances = forest.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    
                    
                    # =============================================================================
                    # Data split
                    # =============================================================================
                    
                    
                    #for f_Num_ in [int(f_Num*0.6)]:
                    for f_Num_ in [200, 160, 100, 80]:
                        #print("Data Num: %d "%(f_Num_))
                        feature_sel=[]
                        
                        for lll in range(f_Num_):
                            feature_sel.append(indices[lll])
                        
                        #feature_sel = [d1, d2]
                        #print("feature selected:%s"%(str(feature_sel)))
                        X = M_list[:, feature_sel]#importance TOP5
                        
                        test_prop = 0.3
                        #random_seed = 1
                        
                        
                        ## ロジスティック回帰
                        rand_nn = 10*2
                        acc_test_for_ave = np.zeros([rand_nn, 9])
                        acc_train_for_ave = np.zeros([rand_nn, 9])
                
                        for random_seed_ in range(rand_nn):
                            #
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_prop, random_state=random_seed_)
                            sc = StandardScaler()
                            sc.fit(X_train)
                            X_train_std = sc.transform(X_train)
                            X_test_std = sc.transform(X_test)
                            
                            # K-NN
                            acc_knn_lis_test = []
                            acc_knn_lis_train = []
                            #for k_n in np.arange(1,C):
                            for c in np.arange(-3, 6, dtype=np.float):
                                ##knn = KNeighborsClassifier(n_neighbors = k_n)
                                lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
                            
                                # k近傍法のモデルにトレーニングデータを適合
                                ##knn.fit(X_train_std, y_train)
                                lr.fit(X_train_std, y_train)
                                
                                acc_train = accuracy_score(y_train, lr.predict(X_train_std))
                                acc_test  = accuracy_score(y_test, lr.predict(X_test_std))
                                #
                            
                                acc_knn_lis_test.append(acc_test)
                                acc_knn_lis_train.append(acc_train)
                                
                            acc_test_for_ave[random_seed_ , :] = acc_knn_lis_test
                            acc_train_for_ave[random_seed_ , :] = acc_knn_lis_train
                            
                        acc_test_ave = np.mean(acc_test_for_ave, 0)
                        acc_train_ave = np.mean(acc_train_for_ave, 0)
                        
                        #print (', '.join(['{: .4f}'.format(xx) for xx in acc_tran_ave]))
                        #"feature selected num", "mooving_ave", "num of FFT", "shift", "mfcc_n", 
                        cond = str(f_Num_) +","+ str(num) +","+ str(n_fft_) +","+ str(shift) +","+ str(n) +","
                        res = cond + ', '.join(['{: .4f}'.format(xx) for xx in acc_test_ave])
                        #print(res)
                        fp = open(filename_for_res,"a")
                        fp.write(res+'\n')
                        fp.close()
                    
                    """    
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
                     
                        # =============================================================================
                        # 交差検証 Dicision Tree
                        # =============================================================================
                        #tree = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=1)
                        scores = cross_val_score(tree, X, y, cv=10)
                        print('Cross-Validation scores: {}'.format(scores))
                        print('Average score: {}'.format(np.mean(scores)))
                    """
                    
                    
                    elapsed_time = time.time() - start
                    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
                    
                    
                    """
                    # =============================================================================
                    # SBS K-NN
                    # =============================================================================
                    
                    start2 = time.time()
                    
                    
                    
                    knn = KNeighborsClassifier(n_neighbors=25)
                    
                    # selecting features
                    random_n = 10
                    k_feat_min = 1
                    k_feat_att = 60
                    scores_lis = np.zeros([random_n, f_Num - k_feat_min + 1])
                    subset_lis = np.zeros([random_n, k_feat_att])
                    
                    # SBS
                    for random_state_ in range(random_n):
                        sbs = SBS(knn, k_features=k_feat_min, random_state=random_state_)
                        sbs.fit(X_train_std, y_train)
                        scores_lis[random_state_,:] = np.array(sbs.scores_).reshape(1, len(sbs.scores_))
                        subset_lis[random_state_,:] = np.array(list(sbs.subsets_[ f_Num - k_feat_att ]))
                    
                    scores_lis_mean = np.mean(scores_lis,0)
                    
                    #list marge
                    subset_mrg=[]
                    for jjj in range(len(subset_lis)):
                        list1 = list(subset_lis[jjj])
                        subset_mrg.extend(list1)
                        subset_mrg=list(set(subset_mrg))
                    
                    subset_cnt_list=[]
                    
                    for kkk in range(len(subset_mrg)):
                        subset_cnt=0
                        
                        for iii in range(random_n):
                            for jjj in range(k_feat_att):
                                if subset_lis[iii][jjj] == subset_mrg[kkk]:
                                    subset_cnt = subset_cnt + 1
                                else:
                                    pass
                                    
                        subset_cnt_list.append(subset_cnt)
                    subset_cnt_ =np.vstack([subset_mrg, subset_cnt_list])
                    
                    
                    # plotting performance of feature subsets
                    k_feat = [len(k) for k in sbs.subsets_]
                    plt.figure(1),plt.plot( k_feat, scores_lis_mean, marker='o', label=str(num) )
                    plt.ylabel('Accuracy',fontsize=18)
                    plt.xlabel('Number of features',fontsize=18)
                    plt.grid()
                    plt.tight_layout()
                    plt.legend()
                    # plt.savefig('images/04_08.png', dpi=300)
                    
                    ##plt.figure(2),plt.bar(subset_cnt_[0],subset_cnt_[1])
                    
                    plt.show()
                    
                    elapsed_time2 = time.time() - start2
                    print ("elapsed_time2:{0}".format(elapsed_time2) + "[sec]")
                    
                    #df=pd.DataFrame(subset_cnt_list, index=subset_mrg.round().astype(int))
                    df=pd.DataFrame(subset_cnt_list, index=subset_mrg)
                    df = df.sort_values(by=0,ascending=False)
                    
                    #df.index ->
                    # [20,34,,7,17,14,11,19,30,0,53,5,68,10,60,66,65,29,44,49,59,32,54,1,42,12,7,43,4,25,36,28,84,6,16,48,15,2,75,76,63,80,39,3,31,21,46,79,91,50,78,88,26,22,45,35,37,24,9,13,96,73,93,77,99,41,8,61,67,71,83,47,51,81,62,23,56,94,52,33,82,97,18,70,40,74,72,85,27,69,86,98,58,38,95,55,64,89,87,90]
                    
                    f_best = list(df.index)[0:60]
                    
                    f_best_ = []
                    for iii in f_best:
                        f_best_.append(round(iii))
                    
                    X_ = X[:,f_best_]
                    """
            
"""
test_prop = 0.3
random_seed = 1

X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y, test_size=test_prop, random_state=random_seed)
sc = StandardScaler()
sc.fit(X_train_)
X_train_std = sc.transform(X_train_)
X_test_std = sc.transform(X_test_)

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
"""