import numpy as np
import xgboost as xgb
import sklearn,re,tqdm,glob
from sklearn.model_selection import KFold
import lightgbm as lgb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.fftpack as scifft
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

NFFT = 1024
SEG_LEN = 150000
with open('train.csv','r') as f:
    full_train_dat = pd.read_csv(f)
print('CSV loaded')
NUM_TRAIN_SEGS = int(len(full_train_dat)/SEG_LEN)


test_ids = []    
patt = re.compile('dat/(.*).csv')
full_test_dat = []
for i in tqdm.tqdm(sorted(glob.glob('dat/*'))):
    with open(i,'r') as f:
        full_test_dat.append(pd.read_csv(f))
    test_ids.append(patt.search(i).group(1))
full_test_dat = pd.concat(full_test_dat)
print('Testing data completed')
NUM_TEST_SEGS = len(test_ids)


nfft=1024
nfilt = 24
bin = np.floor(np.linspace(0,nfft/2-2,nfilt+2))
fbank = np.zeros([nfilt,nfft//2-1])
for j in range(0,nfilt):
    for i in range(int(bin[j]), int(bin[j+1])):
        fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
    for i in range(int(bin[j+1]), int(bin[j+2])):
        fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])


def get_features(dat_frame,orig_dat_frame,seglen=150000,labels=None):
    for S in tqdm.tqdm(range(int(len(orig_dat_frame)/seglen))):
        seg = orig_dat_frame.iloc[S*seglen:(S+1)*seglen]
        xdat = pd.Series(seg['acoustic_data'].values)
        if labels is not None:
            labels.loc[S,'label'] = seg['time_to_failure'].values[-1]
        dat_frame.loc[S,'mean'] = xdat.mean()
        dat_frame.loc[S,'std'] = xdat.std()
        dat_frame.loc[S,'min'] = xdat.min()
        dat_frame.loc[S,'max'] = xdat.max()
        dat_frame.loc[S,'amax'] = xdat.abs().max()
        dat_frame.loc[S,'amin'] = xdat.abs().min()
        dat_frame.loc[S,'meanchange'] = np.mean(np.diff(xdat))
        dat_frame.loc[S,'iqr'] = np.subtract(*np.percentile(xdat,[72,25]))
        fftpow = np.abs(np.fft.fft(xdat[:,None],NFFT,0)**2)[1:int(NFFT/2),:]/seglen
        dat_frame.loc[S,'centroid'] = (np.sum((np.arange(fftpow.shape[0])+1)[:,None]*fftpow,0)/np.sum(fftpow,0))[:,None]
        FBEs = np.matmul(fbank,fftpow).squeeze()
        for I,i in enumerate(FBEs):
            dat_frame.loc[S,'fbankenergies_'+str(I)] = i 
        cepstrum = scifft.dct(np.log10(FBEs))
        for I,i in enumerate(cepstrum):
            dat_frame.loc[S,'cepstrum_'+str(I)] = i 
        dat_frame.loc[S,'energy'] = np.sum(fftpow)
        FBCs = (np.sum((np.arange(fftpow.shape[0])[None,None,:]+1)*(fbank*fftpow.T[:,None,:]),2)/np.sum(fbank*fftpow.T[:,None,:],2)).squeeze()
        for I,i in enumerate(FBCs):
            dat_frame.loc[S,'fbankcentroids_'+str(I)] = i 
        for percent in [0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99]:
            dat_frame.loc[S,'aQ'+str(percent)] = np.percentile(np.abs(xdat),percent*100)
            dat_frame.loc[S,'Q'+str(percent)] = np.percentile(xdat,percent*100)
        for windows in [10,100,1000]:
            x_roll_std = xdat.rolling(windows).std().dropna().values
            x_roll_mean = xdat.rolling(windows).mean().dropna().values
            dat_frame.loc[S,'rollstd_mean_w'+str(windows)] = x_roll_std.mean()
            dat_frame.loc[S,'rollstd_std_w'+str(windows)] = x_roll_std.std()
            dat_frame.loc[S,'rollstd_min_w'+str(windows)] = x_roll_std.min()
            dat_frame.loc[S,'rollstd_max_w'+str(windows)] = x_roll_std.max()
            dat_frame.loc[S,'rollstd_amax_w'+str(windows)] = np.abs(x_roll_std).max()
            dat_frame.loc[S,'rollstd_amin_w'+str(windows)] = np.abs(x_roll_std).min()
            dat_frame.loc[S,'rollstd_stdchange_w'+str(windows)] = np.mean(np.diff(x_roll_std))
            for percent in [0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99]:
                dat_frame.loc[S,'rollstd_Q'+str(percent)+'_w'+str(windows)] = np.percentile(x_roll_std,percent*100)
            dat_frame.loc[S,'rollmean_mean_w'+str(windows)] = x_roll_mean.mean()
            dat_frame.loc[S,'rollmean_std_w'+str(windows)] = x_roll_mean.std()
            dat_frame.loc[S,'rollmean_min_w'+str(windows)] = x_roll_mean.min()
            dat_frame.loc[S,'rollmean_max_w'+str(windows)] = x_roll_mean.max()
            dat_frame.loc[S,'rollmean_amax_w'+str(windows)] = np.abs(x_roll_mean).max()
            dat_frame.loc[S,'rollmean_amin_w'+str(windows)] = np.abs(x_roll_mean).min()
            dat_frame.loc[S,'rollmean_meanchange_w'+str(windows)] = np.mean(np.diff(x_roll_mean))
            for percent in [0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99]:
                dat_frame.loc[S,'rollmean_Q'+str(percent)+'_w'+str(windows)] = np.percentile(x_roll_mean,percent*100)
    if labels is not None:
        return dat_frame,labels
    else:
        return dat_frame
#train_dat = get_features(train_dat,full_train_dat,labels=True)
#test_dat = get_features(test_dat,full_test_dat,labels=False)

X_tr = pd.DataFrame(index=range(NUM_TRAIN_SEGS), dtype=np.float64)
X_te = pd.DataFrame(index=range(NUM_TEST_SEGS), dtype=np.float64)
Y_tr = pd.DataFrame(index=range(NUM_TEST_SEGS), dtype=np.float64)

X_tr,Y_tr = get_features(X_tr,full_train_dat,labels=Y_tr)
X_te = get_features(X_te,full_test_dat)

scaler = StandardScaler()
scaler.fit(X_tr)
X_train_scaled = pd.DataFrame(scaler.transform(X_tr), columns=X_tr.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns)





n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=7)



# XGBOOST-----------------------------------------------------------------------


oof = np.zeros(len(X_train_scaled))
prediction = [np.zeros(len(X_test_scaled))]
scores = []
feature_importance = pd.DataFrame()
eta_vals = [0.003,0.01,0.03]
sub_vals = [0.5,0.9,1]
depth_vals = [4,5,6,7,8,9]
min_score = np.inf
prediction = [np.zeros(len(X_test_scaled))]
for eta in eta_vals:
    for subsample in sub_vals:
        for max_depth in depth_vals:
            params = {'eta': eta,
                  'max_depth': max_depth,
                  'subsample': subsample,
                  'objective': 'reg:linear',
                  'eval_metric': 'mae',
                  'silent': True,
                  'gpu_id':0,
                  'tree_method':'gpu_hist',
                  'nthread': 4}
            oof = np.zeros(len(X_train_scaled))
            tmp_prediction = np.zeros(len(X_test_scaled))
            scores = []
            feature_importance = pd.DataFrame()
            for fold_n, (train_index,valid_index) in enumerate(folds.split(X_train_scaled)):
                print('Fold %i started...'%(fold_n+1))
                X_train, X_valid = X_train_scaled.iloc[train_index], X_train_scaled.iloc[valid_index]
                Y_train, Y_valid = Y_tr.iloc[train_index],Y_tr.iloc[valid_index]
                X_train = X_train.loc[:,X_train.columns!='label']
                X_valid = X_valid.loc[:,X_valid.columns!='label']
                X_test = X_test_scaled
                train_data = xgb.DMatrix(data=X_train, label=Y_train, feature_names=X_train.columns)
                valid_data = xgb.DMatrix(data=X_valid, label=Y_valid, feature_names=X_train.columns)
                watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
                model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)
                y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X_train.columns), ntree_limit=model.best_ntree_limit)
                y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_train.columns), ntree_limit=model.best_ntree_limit)
                oof[valid_index] = y_pred_valid.reshape(-1,)
                scores.append(mean_absolute_error(Y_valid,y_pred_valid))
                tmp_prediction += y_pred
                #Feature importance
                fold_importance = pd.DataFrame()
                fold_importance["feature"] = X_train.columns
                feature_importances = model.get_fscore()
                fold_importance["importance"] = [feature_importances[i] if i in feature_importances.keys() else 0 for i in X_train.columns]
                fold_importance["fold"] = fold_n + 1
                feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
            scores_fold = np.mean(scores)
            if scores_fold < min_score:
                min_score = scores_fold 
                best_params = params
                prediction[-1] = tmp_prediction
                save_feature_importance = feature_importance
                save_oof = oof

feature_importance = save_feature_importance
oof = save_oof
print('Best parameters for XGBoost...')
print(best_params)

prediction[-1] /= float(n_fold)
feature_importance["importance"] /= float(n_fold)
cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

plt.figure(1,figsize=(16, 12));
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
plt.title('XGboost Features (avg over folds)');
        
    

# LIGHTGBM----------------------------------------------------------------------
oof2 = np.zeros(len(X_train_scaled))
prediction.append(np.zeros(len(X_test_scaled)))
scores2 = []
feature_importance2 = pd.DataFrame()
for fold_n, (train_index,valid_index) in enumerate(folds.split(X_train_scaled)):
    X_train, X_valid = X_train_scaled.iloc[train_index], X_train_scaled.iloc[valid_index]
    Y_train, Y_valid = Y_tr.iloc[train_index],Y_tr.iloc[valid_index]
    X_train = X_train.loc[:,X_train.columns!='label']
    X_valid = X_valid.loc[:,X_valid.columns!='label']
    X_test = X_test_scaled
    params = {'num_leaves': 54,
          'min_data_in_leaf': 79,
          'objective': 'huber',
          'max_depth': -1,
          'learning_rate': 0.01,
          "boosting": "gbdt",
          "bagging_freq": 5,
          "bagging_fraction": 0.8126672064208567,
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1,
          'reg_alpha': 0.1302650970728192,
          'reg_lambda': 0.3603427518866501,
         }
    print('Fold %i started...'%(fold_n))            
    model = lgb.LGBMRegressor(n_estimators = 50000, n_jobs = -1,**params)
    model.fit(X_train, Y_train, 
            eval_set=[(X_train, Y_train), (X_valid, Y_valid)], eval_metric='mae',
            verbose=10000, early_stopping_rounds=200)
    
    y_pred_valid = model.predict(X_valid)
    y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
    prediction[-1] += y_pred
    oof2[valid_index] = y_pred_valid.reshape(-1,)
    fold_importance = pd.DataFrame()
    fold_importance["feature"] = X_train.columns
    fold_importance["importance"] = model.feature_importances_
    fold_importance["fold"] = fold_n + 1
    feature_importance2 = pd.concat([feature_importance2, fold_importance], axis=0)
    scores2.append(mean_absolute_error(Y_valid,y_pred_valid))

prediction[-1] /= float(n_fold)
feature_importance2["importance"] /= float(n_fold)
cols = feature_importance2[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

best_features = feature_importance2.loc[feature_importance2.feature.isin(cols)]

plt.figure(2,figsize=(16, 12));
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
plt.title('LGB Features (avg over folds)');
        


end_prediction = np.mean(prediction,0)
import datetime
fname='submission_'+str(datetime.datetime.now())[:19].replace(" ","_").replace(":","-")+'.csv'
print('Saving to file submissions/'+fname)
with open('submissions/'+fname,'w') as f:
    f.write("seg_id,time_to_failure\n")
    for I,i in enumerate(test_ids):
        f.write('%s,%1.4f\n'%(i,end_prediction[I]))



































