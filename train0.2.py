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
from sklearn.svm import NuSVR, SVR

NFFT = 1024
SEG_LEN = 150000
with open('../lanl_earthquake_prediction/train.csv','r') as f:
    full_train_dat = pd.read_csv(f)
print('CSV loaded')
NUM_TRAIN_SEGS = int(len(full_train_dat)/SEG_LEN)


test_ids = []    
patt = re.compile('../lanl_earthquake_prediction/dat/(.*).csv')
full_test_dat = []
for i in tqdm.tqdm(sorted(glob.glob('../lanl_earthquake_prediction/dat/*'))):
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
        #dat_frame.loc[S,'amin'] = xdat.abs().min()
        dat_frame.loc[S,'meanchange'] = np.mean(np.diff(xdat))
        dat_frame.loc[S,'mad'] = xdat.mad()
        dat_frame.loc[S,'kurtosis'] = xdat.kurtosis()
        dat_frame.loc[S,'skew'] = xdat.skew()
        dat_frame.loc[S,'iqr'] = np.subtract(*np.percentile(xdat,[75,25]))
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
            #dat_frame.loc[S,'aQ'+str(percent)] = np.percentile(np.abs(xdat),percent*100)
            dat_frame.loc[S,'Q'+str(percent)] = np.percentile(xdat,percent*100)
        for windows in [10,100,1000]:

            x_roll_std = xdat.rolling(windows).std().dropna().values
            dat_frame.loc[S,'rollstd_mean_w'+str(windows)] = x_roll_std.mean()
            dat_frame.loc[S,'rollstd_std_w'+str(windows)] = x_roll_std.std()
            dat_frame.loc[S,'rollstd_min_w'+str(windows)] = x_roll_std.min()
            dat_frame.loc[S,'rollstd_max_w'+str(windows)] = x_roll_std.max()
            dat_frame.loc[S,'rollstd_amax_w'+str(windows)] = np.abs(x_roll_std).max()
            dat_frame.loc[S,'rollstd_amin_w'+str(windows)] = np.abs(x_roll_std).min()
            dat_frame.loc[S,'rollstd_meanchange_w'+str(windows)] = np.mean(np.diff(x_roll_std))
            for percent in [0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99]:
                dat_frame.loc[S,'rollstd_Q'+str(percent)+'_w'+str(windows)] = np.percentile(x_roll_std,percent*100)

            x_roll_mean = xdat.rolling(windows).mean().dropna().values
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
Y_tr = pd.DataFrame(index=range(NUM_TRAIN_SEGS), dtype=np.float64)

X_tr,Y_tr = get_features(X_tr,full_train_dat,labels=Y_tr)
X_te = get_features(X_te,full_test_dat)

scaler = StandardScaler()
scaler.fit(X_tr)
X_train_scaled = pd.DataFrame(scaler.transform(X_tr), columns=X_tr.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns)





n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=7)
oof = [np.zeros(len(X_train_scaled))]
prediction = [np.zeros(len(X_test_scaled))]

#-------------------------------------------------------------------------------
# XGBOOST-----------------------------------------------------------------------
#-------------------------------------------------------------------------------
scores = []
feature_importance = pd.DataFrame()
eta_vals = [0.003]
sub_vals = [0.75]
depth_vals = [5]
min_score = np.inf
prediction = [np.zeros(len(X_test_scaled))]
grid_search_results = [[[],[],[]]]
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
            grid_search_results.append([params,np.zeros(len(X_train_scaled)),np.zeros(len(X_test_scaled))])
            scores = []
            feature_importance = pd.DataFrame()
            print('Starting to train model with params:')
            print(params)
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
                grid_search_results[-1][1][valid_index] = y_pred_valid.reshape(-1,)
                scores.append(mean_absolute_error(Y_valid,y_pred_valid))
                grid_search_results[-1][2] += y_pred
                #Feature importance
                fold_importance = pd.DataFrame()
                fold_importance["feature"] = X_train.columns
                feature_importances = model.get_fscore()
                fold_importance["importance"] = [feature_importances[i] if i in feature_importances.keys() else 0 for i in X_train.columns]
                fold_importance["fold"] = fold_n + 1
                feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
            scores_fold = np.mean(scores)
            grid_search_results[-1][2] /= float(n_fold)
            if scores_fold < min_score:
                min_score = scores_fold 
                best_params = grid_search_results[-1][0]
                prediction[-1] = grid_search_results[-1][2]
                save_feature_importance = feature_importance
                oof[-1] = grid_search_results[-1][1]

feature_importance = save_feature_importance
print('Best parameters for XGBoost...')
print(best_params)

feature_importance["importance"] /= float(n_fold)
cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

plt.figure(1);
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
plt.title('XGboost Features (avg over folds)');
        
    
#-------------------------------------------------------------------------------
# LIGHTGBM----------------------------------------------------------------------
#-------------------------------------------------------------------------------
oof.append(np.zeros(len(X_train_scaled)))
grid_search_results2 = []
min_score2 = np.inf
prediction.append(np.zeros(len(X_test_scaled)))
leaves_vals = [50,100,150,200]
mindatleaf_vals = [1,32,64,128]
regalpha_vals = [0,0.1,0.2,0.3]
reglambda_vals = [0.,0.1,0.2,0.3]
for leaves in leaves_vals:
    for mindatleaf in mindatleaf_vals:
        for alpha in regalpha_vals:
            for lambdav in reglambda_vals:
                params2 = {'num_leaves': leaves,
                      'min_data_in_leaf': mindatleaf,
                      'max_depth': -1,
                      'learning_rate': 0.01,
                      "boosting": "gbdt",
                      "bagging_freq": 5,
                      "bagging_fraction": 0.8,
                      "bagging_seed": 11,
                      "metric": 'mae',
                      "verbose": -1,
                      'reg_alpha': 0.2,#alpha,
                      'reg_lambda': 0.3,#lambdav,
                      'device':'gpu',
                      'gpu_device_id':0,
                      'gpu_platform_id':0
                     }
                print('Starting to train model with params:')
                print(params2)
                feature_importance2 = pd.DataFrame()
                scores2_fold = []
                grid_search_results2.append([params2,np.zeros(len(X_train_scaled)),np.zeros(len(X_test_scaled))])
                for fold_n, (train_index,valid_index) in enumerate(folds.split(X_train_scaled)):
                    X_train, X_valid = X_train_scaled.iloc[train_index], X_train_scaled.iloc[valid_index]
                    Y_train, Y_valid = Y_tr.iloc[train_index],Y_tr.iloc[valid_index]
                    X_train = X_train.loc[:,X_train.columns!='label']
                    X_valid = X_valid.loc[:,X_valid.columns!='label']
                    X_test = X_test_scaled
                    print('Fold %i started...'%(fold_n))            
                    model = lgb.LGBMRegressor(n_estimators = 50000, n_jobs = -1,**params2)
                    model.fit(X_train, Y_train, 
                            eval_set=[(X_train, Y_train), (X_valid, Y_valid)], eval_metric='mae',
                            verbose=10000, early_stopping_rounds=200)
                    
                    y_pred_valid = model.predict(X_valid)
                    y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
                    grid_search_results2[-1][2] += y_pred
                    grid_search_results2[-1][1][valid_index] = y_pred_valid.reshape(-1,)
                    fold_importance = pd.DataFrame()
                    fold_importance["feature"] = X_train.columns
                    fold_importance["importance"] = model.feature_importances_
                    fold_importance["fold"] = fold_n + 1
                    feature_importance2 = pd.concat([feature_importance2, fold_importance], axis=0)
                    scores2_fold.append(mean_absolute_error(Y_valid,y_pred_valid))
                grid_search_results2[-1][2] /= float(n_fold)
                scores2_total = np.mean(scores2_fold)
                if scores2_total < min_score2:
                    min_score2 = scores2_total 
                    best_params2 = grid_search_results2[-1][0]
                    oof[-1] = grid_search_results2[-1][1]
                    prediction[-1] = grid_search_results2[-1][2]
                    save_feature_importance = feature_importance2
print('Best parameters for LightGBM...')
print(best_params2)

feature_importance2 = save_feature_importance
feature_importance2["importance"] /= float(n_fold)
cols = feature_importance2[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

best_features = feature_importance2.loc[feature_importance2.feature.isin(cols)]

plt.figure(2,figsize=(16, 12));
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
plt.title('LGB Features (avg over folds)');
        


#-------------------------------------------------------------------------------
#NuSVM--------------------------------------------------------------------------
#-------------------------------------------------------------------------------
nuvals = [0.85,0.9,0.95]
Cvals = [1.,2.,3.]
prediction.append(np.zeros(len(X_test_scaled)))
oof.append(np.zeros(len(X_train_scaled)))
grid_search_results3 = []
min_score3 = np.inf
for nu in nuvals:
    for C in Cvals:
        grid_search_results3.append([{'nu':nu,'C':C},np.zeros(len(X_train_scaled)),np.zeros(len(X_test_scaled))])
        scores3_fold = []
        print('Training model with')
        print(grid_search_results3[-1][0])
        for fold_n, (train_index,valid_index) in enumerate(folds.split(X_train_scaled)):
            model = NuSVR(gamma='scale', nu=nu, C=C, tol=0.01)
            X_train, X_valid = X_train_scaled.iloc[train_index], X_train_scaled.iloc[valid_index]
            Y_train, Y_valid = Y_tr.iloc[train_index].squeeze(),Y_tr.iloc[valid_index].squeeze()
            X_train = X_train.loc[:,X_train.columns!='label']
            X_valid = X_valid.loc[:,X_valid.columns!='label']
            X_test = X_test_scaled
            model.fit(X_train,Y_train)
            Y_pred_valid = model.predict(X_valid).reshape(-1,)
            scores3_fold.append(mean_absolute_error(Y_valid, Y_pred_valid))
            print('Fold {0}. MAE: {1}.'.format(fold_n,scores3_fold[-1]))
            grid_search_results3[-1][1][valid_index] = Y_pred_valid            
            y_pred = model.predict(X_test).reshape(-1,)
            grid_search_results3[-1][2] += y_pred
        scores3_total = np.mean(scores3_fold)
        grid_search_results3[-1][2] /= n_fold
        if scores3_total < min_score3:
            min_score3 = scores3_total 
            best_params3 = grid_search_results3[-1][0]
            oof[-1] = grid_search_results3[-1][1]
            prediction[-1] = grid_search_results3[-1][2]
print('Best parameters for NuSVM...')
print(best_params3)

    












#-------------------------------------------------------------------------------
#Write File---------------------------------------------------------------------
#-------------------------------------------------------------------------------

end_prediction = np.mean(prediction,0)
import datetime
fname='submission_'+str(datetime.datetime.now())[:19].replace(" ","_").replace(":","-")+'.csv'
print('Training error from cross validation:')
print(np.mean(np.abs(np.mean(oof,0) - Y_tr.values[:,0])))
print('Saving to file submissions/'+fname)
with open('submissions/'+fname,'w') as f:
    f.write("seg_id,time_to_failure\n")
    for I,i in enumerate(test_ids):
        f.write('%s,%1.4f\n'%(i,end_prediction[I]))



































