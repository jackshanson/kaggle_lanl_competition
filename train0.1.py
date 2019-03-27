import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import glob,re,tqdm 
import numpy as np
import datetime


def get_feats(dat):
    feats = {}
    feats['max'] = [np.max(dat)]
    feats['min'] = [np.min(dat)]
    feats['ave'] = [np.mean(dat)]
    feats['std'] = [np.std(dat)]
    feats['Q'] = np.percentile(dat,[1,10,25,50,75,95])
    feats['amax'] = [np.max(np.abs(dat))]
    feats['amin'] = [np.min(np.abs(dat))]
    feats['adiff'] = [np.mean(np.diff(dat))]
    feats['pdiff'] = (np.diff(dat)/dat[0:-1])
    feats['pdiff'][np.isnan(feats['pdiff'])] = 0
    feats['pdiff'][feats['pdiff']==-np.inf] = -1
    feats['pdiff'][feats['pdiff']==np.inf] = 1
    feats['pdiff'] = [np.mean(feats['pdiff'])]
    feats['AQ'] = np.percentile(np.abs(dat),[1,10,25,50,75,95])
    for window in [10,100,1000]:
        rolling_mean = dat.rolling(window).mean().dropna().values
        rolling_std  = dat.rolling(window).std().dropna().values
        rolling_max  = dat.rolling(window).max().dropna().values
        rolling_min  = dat.rolling(window).min().dropna().values
        feats['stdmax'+str(window)] = [np.max(rolling_std)]
        feats['stdmin'+str(window)] = [np.min(rolling_std)]
        feats['stdave'+str(window)] = [np.mean(rolling_std)]
        feats['stdstd'+str(window)] = [np.std(rolling_std)]
        feats['stdQ'+str(window)] = np.percentile(rolling_std,[1,10,25,50,75,95])
        feats['adiffstd'+str(window)] = [np.mean(np.diff(rolling_std))]
        feats['mumax'+str(window)] = [np.max(rolling_mean)]
        feats['mumin'+str(window)] = [np.min(rolling_mean)]
        feats['muave'+str(window)] = [np.mean(rolling_mean)]
        feats['mustd'+str(window)] = [np.std(rolling_mean)]
        feats['muQ'+str(window)] = np.percentile(rolling_mean,[1,10,25,50,75,95])
        feats['adiffmu'+str(window)] = [np.mean(np.diff(rolling_mean))]
    return feats


SAMPLELEN = 150000
OVERLAP = 1
data ={}
with open('train.csv','r') as f:
    train_dat = pd.read_csv(f)
print('CSV loaded')
for i in tqdm.tqdm(range(int(len(train_dat)/SAMPLELEN/OVERLAP))):
    data[i] = [get_feats(train_dat.acoustic_data[i*int(SAMPLELEN*OVERLAP):(i+1)*int(SAMPLELEN*OVERLAP)]),train_dat.time_to_failure[(i+1)*int(SAMPLELEN*OVERLAP)]]
print('Training data completed')
train_ids = range(int(len(train_dat)/SAMPLELEN/OVERLAP*0.9))
val_ids = range(int(len(train_dat)/SAMPLELEN/OVERLAP*0.9),int(len(train_dat)/SAMPLELEN/OVERLAP))
test_ids = []    
patt = re.compile('dat/(.*).csv')
for i in tqdm.tqdm(sorted(glob.glob('dat/*'))):
    with open(i,'r') as f:
        data[patt.search(i).group(1)] = [get_feats(pd.read_csv(f).acoustic_data),-1]
    test_ids.append(patt.search(i).group(1))
print('Testing data completed')


feats_table = [['stdmax100',1],
['stdmin100',1],
['adiff',1],
['ave',1],
['muQ10',5],
['muave1000',1],
['stdmax10',1],
['stdmax1000',1],
['adiffstd10',1],
['mustd1000',1],
['adiffmu100',1],
['min',1],
['stdstd1000',1],
['muQ1000',5],
['mumax100',1],
['stdstd10',1],
['stdave10',1],
['stdQ1000',5],
['stdmin1000',1],
['stdQ10',5],
['mumin10',1],
['adiffstd100',1],
['stdQ100',5],
['pdiff',1],
['max',1],
['mumax1000',1],
['mumax10',1],
['stdstd100',1],
['AQ',5],
['mumin100',1],
['stdave1000',1],
['mustd100',1],
['muave10',1],
['amax',1],
['std',1],
['Q',5],
['muQ100',5],
['amin',1],
['adiffstd1000',1],
['muave100',1],
['mumin1000',1],
['stdmin10',1],
['stdave100',1],
['adiffmu1000',1],
['adiffmu10',1],
['mustd10',1],
]

cum_feat_size = np.cumsum([0]+[i[1] for i in feats_table])


train_dat = np.array([np.concatenate([data[i][0][j[0]] for j in feats_table]) for i in train_ids])
train_lab = np.array([data[i][1] for i in train_ids])[:,None]
Dtrain = xgb.DMatrix(train_dat,label=train_lab)

val_dat = np.array([np.concatenate([data[i][0][j[0]] for j in feats_table]) for i in val_ids])
val_lab = np.array([data[i][1] for i in val_ids])[:,None]
Dval = xgb.DMatrix(val_dat,label=val_lab)

test_dat = np.array([np.concatenate([data[i][0][j[0]] for j in feats_table]) for i in test_ids])
Dtest = xgb.DMatrix(test_dat)


val_pred= [] 
test_pred = []
val_metrics = []
model_params = []
feature_scores = []
min_mae = np.inf

depthvals = range(1,5)
etavals =[0.001,0.003,0.01,0.03]
binvals = [256]
#estimatorvals = [16,32,64,96]
childweightvals = [1,1000,10000]
subsamplevals = [0.1,0.3,0.5,0.75,0.9,1.]
samplebytreevals = [0.3,0.5,0.75,0.9,1.]
total_num =len(depthvals)*len(etavals)*len(binvals)*len(childweightvals)*len(subsamplevals)*len(samplebytreevals)
progress =0
print('')
for _max_depth in depthvals:
    for _eta in etavals:
        for _max_bin in binvals:
            for _min_child_weight in childweightvals:
                for _subsample in subsamplevals:
                    for _colsample_bytree in samplebytreevals:
                        #for _n_estimators in estimatorvals:
                        params = {
                            # Parameters that we are going to tune.
                            'max_depth':_max_depth,
                            #'lambda': _lambda,
                            'min_child_weight':_min_child_weight,
                            'learning_rate':_eta,
                            'subsample': _subsample,
                            'colsample_bytree': _colsample_bytree,
                            # Other parameters
                        }
                        params['gpu_id']=0
                        params['tree_method'] = 'gpu_hist'
                        params['max_bin'] = _max_bin
                        params['objective'] = 'reg:linear'
                        params['eval_metric'] = 'mae'
                        #params[]

                        model = xgb.train(params,Dtrain,10000,evals=[[Dval,'Validation']],early_stopping_rounds=200,verbose_eval=False)
                        val_metrics.append(model.best_score)
                        if model.best_score < min_mae:
                            min_mae = model.best_score
                            best_model = progress
                        fscores = model.get_fscore()
                        feature_scores.append([np.mean([fscores['f'+str(j)] if 'f'+str(j) in fscores.keys() else 0 for j in range(cum_feat_size[i],cum_feat_size[i+1])]) for i in range(len(feats_table))])
                        model_params.append([_max_depth,_eta,_max_bin,_min_child_weight,_subsample,_colsample_bytree])
                        val_pred.append(model.predict(Dval,ntree_limit=model.best_ntree_limit))
                        test_pred.append(model.predict(Dtest,ntree_limit=model.best_ntree_limit))
                        progress+=1    
                        print('\r%i/%i (%3.2f%%) Current mae: %1.6f; best mae: %1.6f from model %i'%(progress,total_num,progress/float(total_num)*100,model.best_score,min_mae,best_model)),

best_models = np.argsort(val_metrics)
print('Best model is model %i (MAE:%f), which corresponds to:'%(best_models[0]+1,val_metrics[best_models[0]]))
print(model_params[best_models[0]])
np.mean(np.abs(val_pred[best_models[0]] - val_lab[:,0]))
min_mae = np.inf
optimal_model = 0
for k in range(50):
    mae = np.mean(np.abs(np.mean([val_pred[i] for i in best_models[:k]],0) - val_lab[:,0]))
    if mae<min_mae:
        min_mae = mae
        optimal_model = k
print(min_mae,optimal_model)


test_pred_submit = np.mean([test_pred[i] for i in best_models[:optimal_model]],0)

fname='submission_'+str(datetime.datetime.now())[:19].replace(" ","_").replace(":","-")+'.csv'
print('Saving to file submissions/'+fname)
with open('submissions/'+fname,'w') as f:
    f.write("seg_id,time_to_failure\n")
    for I,i in enumerate(test_ids):
        f.write('%s,%1.4f\n'%(i,test_pred_submit[I]))





    
