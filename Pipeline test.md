

```python
from QSAR_package.data_split import extractData,randomSpliter
from QSAR_package.feature_preprocess import correlationSelection,RFE_ranking
from QSAR_package.data_scale import dataScale
from QSAR_package.grid_search import gridSearchPlus,gridSearchBase
from QSAR_package.model_evaluation import modeling,modelEvaluator
import pandas as pd
import numpy as np
import os,sys
```


```python
parent_dir = os.path.dirname(os.getcwd())
```


```python
os.path.join(parent_dir,'scscore-master')
```

### 输入总描述符CSV文件$\to$随机分集$\to$相关性筛选(排序)$\to$数据压缩 [ $\to$RFE排序 ]


```python
file_name = "C:/Users/buct408a/Desktop/test/spla2_296_rdkit2d.csv"

randx = 12
spliter = randomSpliter(test_size=0.25,random_state=randx)
spliter.ExtractTotalData(file_name,label_name='pIC50')
spliter.SplitData()

# tr_x = spliter.tr_x.loc[:,spliter.tr_x.columns.str.match(r'(?!RDF)')]
tr_x = spliter.tr_x
tr_y = spliter.tr_y
te_y = spliter.te_y

corr = correlationSelection()
corr.PearsonXX(tr_x, tr_y,threshold_low=0.01, threshold_up=0.99)

scaler = dataScale(scale_range=(0.1, 0.9))

tr_scaled_x = scaler.FitTransform(corr.selected_tr_x.iloc[:,:])
te_scaled_x = scaler.Transform(spliter.te_x,DataSet='test')

# rfe = RFE_ranking('SVR',features_num=1)
# rfe.Fit(tr_scaled_x, tr_y)

# tr_ranked_x = rfe.tr_ranked_x
# te_ranked_x = te_scaled_x.loc[:,tr_ranked_x.columns]
print(tr_scaled_x.shape)
```

    (222, 117)
    


```python
len(np.unique(tr_y))
```




    173



### repeat寻优，找出最佳参数 [ 及描述符数量 ]

### 拟合模型，评价模型，保存结果


```python
grid = gridSearchPlus(grid_estimatorName='SVC', fold=3, repeat=1, early_stop=0.01, scoreThreshold=1)
# grid.FitWithFeaturesNum(tr_ranked_x, tr_y,features_range=(5,23))   # 用RFE排序
# grid.FitWithFeaturesNum(tr_scaled_x, tr_y,features_range=(5,7))      # 用pearson相关性排序
grid.Fit(tr_scaled_x,tr_y)  # 不带描述符数量的寻优

### 拟合模型，评价模型，保存结果

model = modeling(grid.best_estimator,params=grid.best_params)
model.Fit(tr_scaled_x.loc[:,grid.best_features], tr_y)
model.Predict(te_scaled_x.loc[:,grid.best_features],te_y)
# model.CrossVal(cv=5)
model.ShowResults(show_cv=False, make_fig=False)
# model.SaveResults(file_name[:-4]+'_results.csv',notes='SVM-RFE,split_seed=120,gridCV=5')
```

    第1/1次gridsearch，此轮耗时00h:00m:05s
    [41;1mscoreThreshold值太高，重新设定为所有打分的0.8分位数0.8055555555555556[0m
    [45m执行early_stop[0m
    1次gridsearch执行完毕，总耗时00h:00m:05s，可通过best_params属性查看最优参数，通过cv_results属性查看所有结果
    Training results: [1m{'accuracy': 1.0, 'mcc': 1.0, 'tp': 29, 'fp': 0, 'tn': 7, 'fn': 0, 'se': 1.0, 'sp': 1.0}[0m
    Test results: [1m{'accuracy': 0.9231, 'mcc': 0.68, 'tp': 11, 'fp': 1, 'tn': 1, 'fn': 0, 'se': 1.0, 'sp': 0.5}[0m
    [1m    accuracy   fn   fp   mcc   se   sp   tn    tp
    tr    1.0000  0.0  0.0  1.00  1.0  1.0  7.0  29.0
    te    0.9231  0.0  1.0  0.68  1.0  0.5  1.0  11.0[0m
    


```python
model.tr_y
```


```python
grid_test = GridSearchCV()
```


```python
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import pickle
```


```python
joblib.dump(model.estimator,file_name[:-4]+'_results.md')
```


```python
rgr = joblib.load("C:/Users/buct408a/Desktop/spla2_296_rdkit2d_results_RandomForestRegressor_118.model")
```

    C:\Anaconda3\lib\site-packages\sklearn\base.py:253: UserWarning: Trying to unpickle estimator DecisionTreeRegressor from version 0.20.1 when using version 0.20.3. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)
    C:\Anaconda3\lib\site-packages\sklearn\base.py:253: UserWarning: Trying to unpickle estimator RandomForestRegressor from version 0.20.1 when using version 0.20.3. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)
    


```python
rgr
```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=28,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=80, n_jobs=None,
               oob_score=False, random_state=0, verbose=0, warm_start=False)




```python
tr_pred_y = rgr.predict(tr_scaled_x)
te_pred_y = rgr.predict(te_scaled_x)
```


```python
tr_eva = modelEvaluator(tr_y, tr_pred_y,model_kind='rgr')
te_eva = modelEvaluator(te_y, te_pred_y,model_kind='rgr')
print(tr_eva.r2,tr_eva.rmse)
print(te_eva.r2,te_eva.rmse)
```

    0.9263 0.3442
    0.5377 0.8584
    


```python
for i in range(*grid.features_range):
    print(i)
```


```python
from sklearn.svm import SVR
import matplotlib.pyplot as plt
```


```python
model = modeling(SVR(),params={'C': 8, 'epsilon': 0.125, 'gamma': 1})
model.Fit(tr_scaled_x.iloc[:,:20], tr_y)
model.Predict(te_scaled_x.iloc[:,:20],te_y)
model.ShowResults(show_cv=False)
```


```python
tr_y = model.tr_y
tr_pred = model.tr_pred_y
te_y = model.te_y
te_pred = model.te_pred_y

fig = plt.figure(figsize=(6,6))
axisMin = min(tr_y.min(),te_y.min(),tr_pred.min(),te_pred.min())-0.5
axisMax = max(tr_y.max(),te_y.max(),tr_pred.max(),te_pred.max())+0.5
plt.plot(tr_y,tr_pred,'xb',markersize=8)
plt.plot(te_y,te_pred,'or',mfc='w',markersize=6)
plt.plot([axisMin,axisMax],[axisMin,axisMax],'k',lw=1)
plt.axis([axisMin,axisMax,axisMin,axisMax])
plt.xlabel('pIC50 values (true)',fontproperties='Times New Roman',fontsize=13)
plt.ylabel('pIC50 values (predicted)',fontproperties='Times New Roman',fontsize=13)
plt.legend(['training set', 'test set'], loc='best')
plt.title('Model 7B',fontproperties='Arial',fontsize=15)
plt.savefig('C:/OneDrive/Jupyter_notebook/regression_new/data/fig/scatter_fig_{}.tif'.format(randx),
            dpi=300,bbox_inches='tight')
plt.show()
```


```python
tr_scaled_x.iloc[:,:20].columns
```


```python
des_0 = tr_scaled_x.iloc[:,:21].columns
len(des_0)
```


```python
des_1 = tr_scaled_x.iloc[:,:20].columns
len(des_1)
```


```python
des_2 = tr_scaled_x.iloc[:,:21].columns
len(des_2)
```


```python
des_3group = pd.DataFrame([des_0,des_1,des_2],index=[0,12,120]).T
```


```python
des_0.difference()
```


```python
des_3group.to_csv('C:/OneDrive/Jupyter_notebook/regression_new/data/descriptors_pearsonRank_0_12_120.csv',index=False)
```


```python
merged_des = pd.DataFrame(des_0.intersection(des_1).intersection(des_2),columns=['Name'])
```


```python
a_list = [1, 2, 3, 4, 5]
b_list = [1, 4, 5]
c_list = [4]
ret_list = list((set(a_list).union(set(b_list)).union(set(c_list)))^(set(a_list)^set(b_list)))
print(ret_list)
```


```python
rank = []
for n in merged_des.Name:
    rank0 = des_0.tolist().index(n)+1
    rank1 = des_1.tolist().index(n)+1
    rank2 = des_2.tolist().index(n)+1
    rank.append([rank0,rank1,rank2])
```


```python
merged_des_rank = pd.concat([merged_des,pd.DataFrame(rank,columns=['rank_0','rank_12','rank_120'])],axis=1)
```


```python
merged_des_rank.to_csv('C:/OneDrive/Jupyter_notebook/regression_new/data/descriptors_merged_pearsonRank_0_12_120.csv',
                       index=False)
```


```python
data = pd.read_csv("C:/OneDrive/Jupyter_notebook/regression_new/data/spla2_296_rdkit2d.csv")
select_df = data.loc[:,merged_des.Name]
```


```python
sum(select_df.iloc[0,:]==0)
```


```python
template_smi = []
for i in range(len(select_df)):
    if sum(select_df.iloc[i,:]==0) == 0:
        template_smi.append(data.iloc[i,0])
```


```python
template_smi
```


```python
from rdkit import Chem
from rdkit.Chem import Draw,AllChem
from rdkit.Chem.Draw import IPythonConsole,SimilarityMaps
```


```python
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
```


```python
mols = [Chem.MolFromSmiles(m) for m in template_smi]
```


```python
mol = mols[4]
ontribs = rdMolDescriptors._CalcTPSAContribs(mol)
fig = SimilarityMaps.GetSimilarityMapFromWeights(mol,[x for x in contribs], colorMap='jet', contourLines=10)
```


```python
contribs = [float(mol.GetAtomWithIdx(i).GetProp('_EState_VSA1')) for i in range(mol.GetNumAtoms())]
fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, contribs, colorMap='jet', contourLines=10)
```


```python
Chem.Descriptors.Chem()
```


```python
data.BalabanJ.T
```


```python
import argparse
```


```python
parser = argparse.ArgumentParser(description='Manual to this script')
```


```python
parser.add_argument('--input','-i', type=str, default=None)
parser.add_argument('--randx', type=int, default=0)
parser.add_argument('--label_name',type=str, default='Activity')
parser.add_argument('--test_size',type=float, default=0.25)
parser.add_argument('--corr_low',type=float, default=0.1)
parser.add_argument('--corr_up',type=float, default=0.9)
parser.add_argument('--ranker',type=str, default='Pearson',choices=['Pearson', 'SVM-RFE', 'RF-RFE'])
parser.add_argument('--algorithm',type=str, default=None, choices=['SVC', 'DTC', 'RFC','SVR','RFR'])
parser.add_argument('--fold', type=int, default=5)
parser.add_argument('--repeat', type=int, default=3)
parser.add_argument('--early_stop', type=float, default=0)
parser.add_argument('--repeat_threshold', type=float, default=None)

```




    _StoreAction(option_strings=['--batch-size'], dest='batch_size', nargs=None, const=None, default=32, type=<class 'int'>, choices=None, help=None, metavar=None)


