import numpy as np
import json
import pandas as pd
import random
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_coverage(lower_bound, upper_bound, y_true, std_length = False):
    out_of_bound = 0
    N = len(y_true) 
    for i in range(N):
        if y_true[i]<lower_bound[i] or y_true[i]>upper_bound[i]:
            out_of_bound+=1  
    if not std_length:
        return {'cvg': 1-out_of_bound/N,
                'length': np.mean([up - low for up, low in zip(upper_bound, lower_bound)]),
                'deviation': np.mean([deviation(ref, low, up) for ref, up, low in zip(y_true, upper_bound, lower_bound) if ref>up or ref<low]) } 
    else:
        lengths = [up - low for up, low in zip(upper_bound, lower_bound)]
        return {'coverage': 1-out_of_bound/N, 'length': np.mean(lengths), 'std_length': np.std(lengths)}

def deviation(ref, low, up):
    if ref > up:
        return ref - up
    else:
        return low - ref

def compute_quantile_residual(y, y_hat, r_hat, alpha):
    N = len(y)
    q_yhat = np.quantile(np.abs(np.array(y)-np.array(y_hat))/r_hat,np.ceil((N+1)*(1-alpha))/N)
    return  q_yhat

def eval_regression(ytrue, ypred):
    return {'r2': round(r2_score(ytrue, ypred), 3),
            'rmse' : round(np.sqrt(mean_squared_error(ytrue, ypred)), 3),
            'mae' : round(mean_absolute_error(ytrue, ypred), 3)}

def separate_features_target(df, nleads=12, nfold=5):
    features_to_drop = ['fname', 'ytrue', 'target']
    pred_cols = [f'ypred_{i+1}' for i in range(nleads*nfold)]
    features_to_drop.extend(pred_cols)
    X = [list(item) for item in np.array(df.drop(features_to_drop, axis=1))]
    r, y, y_hat = list(df['target']), list(df['ytrue']), list(df[pred_cols].mean(axis=1))
    return X, r, y, y_hat

def get_fnames(pids, dmmld=True): 
    if dmmld:
        clinical_data = pd.read_csv('data/SCR-003.Clinical.Data.csv')
    else:
        clinical_data = pd.read_csv('data/SCR-002.Clinical.Data.csv')
    fnames = list(clinical_data[clinical_data.RANDID.isin(pids)].EGREFID)
    return fnames

def split_patients(dmmld=True): 
    if dmmld:
        clinical_data = pd.read_csv('data/SCR-003.Clinical.Data.csv')
        pids = list(set(clinical_data.RANDID))
        random.seed(3)
        random.shuffle(pids)
        return pids[:7], pids[7:14], pids[14:]
    else:
        clinical_data = pd.read_csv('data/SCR-002.Clinical.Data.csv')
        pids = list(set(clinical_data.RANDID))
        random.seed(3)
        random.shuffle(pids)
        return pids[:14], pids[14:]

def aggregate_qt_preds(preds, method = 'global', nfold=5, nleads=12):
    if method == 'global':
        return flatten(preds)
    elif method == 'folds':
        final_preds = [np.mean(tup) for tup in list(zip(*preds))]  
        assert(len(final_preds)==nfold)
        return final_preds
    elif method == 'leads':
        final_preds = [np.mean(lst) for lst in preds]
        assert(len(final_preds)==nleads)
        return final_preds

def convert_to_percent(lst):
    return [100*val for val in lst]

def flatten(lst):
    new = []
    for elem in lst:
        new.extend(elem)
    return new

def str_to_num(lst):
    return [float(val) for val in lst] 

def save_dict_to_json(d, json_path): 
    with open(json_path, 'w') as f: 
        json.dump(d, f, indent = 6) 

def load_model(filename):
    clf = pickle.load(open(filename, 'rb'))
    return clf 

def remove_nan(lst):
    lst_wo_nan = [val for val in lst if str(val)!='nan']
    return lst_wo_nan