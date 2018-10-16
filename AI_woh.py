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
from matplotlib.colors import ListedColormap
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import pickle
from IPython.display import Image

def versiontuple(v):
    return tuple(map(int, (v.split("."))))


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    edgecolor='black',
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')
        
dirnow = os.path.abspath(".")
datadir = "speech_commands_v0.01"
filename_list = "file_name_list_0.txt"
data_num = 50

LABEL = ["one", "two", "tree"]
print("Clasification:%s"%str(LABEL))
feat_method = ["ave", "min", "max"]
print("feat_method:%s"%str(feat_method))

n=13 #the top 13 Mel-frequency cepstral coefficients (MFCCs)
M_list = np.zeros([data_num*len(LABEL), 13*3*len(feat_method)])# 3(mfcc,Δ，ΔΔ)
L_list = np.zeros([data_num*len(LABEL), 1])

for iii in range(len(LABEL)):
    dfl_path = dirnow +"\\"+ datadir +"\\"+ LABEL[iii] +"\\"+ filename_list
    WFL = pd.read_csv( dfl_path , header=None)
    WF_ex = WFL[0:data_num]
    
    for jjj in range(len(WF_ex)):
        df_path =  dirnow +"\\"+ datadir +"\\"+ LABEL[iii] +"\\"+ WF_ex[0][jjj]
        y, sr = librosa.load(df_path, sr=16000)
        
        # MFCC 
        # http://nbviewer.jupyter.org/github/librosa/librosa/blob/master/examples/LibROSA%20demo.ipynb
        # Next, we'll extract the top 13 Mel-frequency cepstral coefficients (MFCCs)
        mfcc        = librosa.feature.mfcc(y=y, n_mfcc=n)
        # Let's pad on the first and second deltas while we're at it
        delta_mfcc  = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
               
        # For future use, we'll stack these together into one matrix
        M = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
        M_ave = np.mean(M, 1).reshape(1, len(M))
        M_max = np.max(M, 1).reshape(1, len(M))
        M_min = np.min(M, 1).reshape(1, len(M))
        M_all = np.hstack([M_ave, M_max, M_min])
    
        M_list[jjj+len(WF_ex)*iii, :] = M_all
        L_list[jjj+len(WF_ex)*iii, :] = int(iii)

#data_set = np.hstack([L_list, M_list])
#np.savetxt('data_set.csv',data_set,delimiter=',')

"""
# =============================================================================
# Random Forest
# =============================================================================
# Random Forests
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=1000)
rf_reg = rf_reg.fit(M_list, L_list)
fti = rf_reg.feature_importances_
"""


# =============================================================================
# KNN
# =============================================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import time

from sklearn import metrics

d1=109 
d2=56
d3=107
d4=42
d5=41

#feature_sel = [d1, d2, d3, d4, d5]
feature_sel = [d1, d2]
print("feature selected:%s"%(str(feature_sel)))
X = M_list[:, feature_sel]#importance TOP5

y = L_list.ravel()
test_prop = 0.3
random_seed = 1
neighbors = 10




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_prop, random_state=random_seed)

start = time.time()
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

"""
knn = KNeighborsClassifier(n_neighbors=neighbors)
knn.fit(X_train_std, y_train)
y_train_pred = knn.predict(X_train_std)
y_test_pred = knn.predict(X_test_std)
print ('Processing Time:={}'.format(time.time()-start)+"[sec]")
print ('Train:accuracy={}'.format(metrics.accuracy_score(y_train, y_train_pred)))
print ('Test:accuracy={}'.format(metrics.accuracy_score(y_test, y_test_pred)))

# KNN 結果をプロットする 
# 注：2次元のみ
# =============================================================================
x1_min, x1_max = X_train_std[:, 0].min() - 0.5, X_train_std[:, 0].max() + 0.5
x2_min, x2_max = X_train_std[:, 1].min() - 0.5, X_train_std[:, 1].max() + 0.5
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                       np.arange(x2_min, x2_max, 0.02))
                       
Z = knn.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
Z = Z.reshape(xx1.shape)

markers = ('s', 'x', 'o', '^', 'v')
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
cmap = ListedColormap(colors[:len(np.unique(y))])

plt.figure(figsize=(10,10))
plt.subplot(211)

plt.contourf(xx1, xx2, Z, alpha=0.5, cmap=cmap)
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())

for idx, cl in enumerate(np.unique(y_train)):
    plt.scatter(x=X_train_std[y_train == cl, 0], y=X_train_std[y_train == cl, 1],
                alpha=0.8, c=cmap(idx),
                marker=markers[idx], label=cl)


plt.xlabel('%d_feature'%(d1))
plt.ylabel('%d_feature'%(d2))
plt.title('train_data')

plt.subplot(212)

plt.contourf(xx1, xx2, Z, alpha=0.5, cmap=cmap)
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())

for idx, cl in enumerate(np.unique(y_test)):
    plt.scatter(x=X_test_std[y_test == cl, 0], y=X_test_std[y_test == cl, 1],
                alpha=0.8, c=cmap(idx),
                marker=markers[idx], label=cl)


plt.xlabel('%d_feature'%(d1))
plt.ylabel('%d_feature'%(d2))
plt.title('test_data')
plt.show()
# =============================================================================
"""

"""
#PCA
pca = PCA(n_components=2)
pca.fit(X)
targets = y
# 分析結果を元にデータセットを主成分に変換する
transformed = pca.fit_transform(X)

# 主成分をプロットする
for label in np.unique(targets):
    plt.scatter(transformed[targets == label, 0],
                transformed[targets == label, 1],)
plt.title('principal component')
plt.xlabel('pc1')
plt.ylabel('pc2')

# 主成分の寄与率を出力する
print('各次元の寄与率: {0}'.format(pca.explained_variance_ratio_))
print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))

# グラフを表示する
plt.show()
"""


#Decisiontree　Classifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=1)
tree.fit(X_train, y_train)

y_test_pred = tree.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_test_pred)
print("accuracy by tree : {}".format(accuracy))


X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, 
                      classifier=tree, test_idx=range(105, 150))

plt.xlabel('%d_feature'%(d1))
plt.ylabel('%d_feature'%(d2))
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/decision_tree_decision.png', dpi=300)
plt.show()



"""
# 交差検証 KNN
sc = StandardScaler()
sc.fit(X)
X_std = sc.transform(X)

knn = KNeighborsClassifier(n_neighbors = neighbors)

scores = cross_val_score(knn, X_std, y, cv=10)
print('Cross-Validation scores: {}'.format(scores))
print('Average score: {}'.format(np.mean(scores)))
"""

"""
# 交差検証 Dicision Tree
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=1)
scores = cross_val_score(tree, X, y, cv=10)
print('Cross-Validation scores: {}'.format(scores))
print('Average score: {}'.format(np.mean(scores)))
"""

"""
# graphviz dotファイル生成
from sklearn.tree import export_graphviz
export_graphviz(tree, 
                out_file='tree.dot', 
                feature_names=['d1', 'd2'])
"""


"""
# モデル保存/ロード
# https://localab.jp/blog/save-and-load-machine-learning-models-in-python-with-scikit-learn/
filename = 'finalized_model.binaryfile'
pickle.dump(tree, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
"""
