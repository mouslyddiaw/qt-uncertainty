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
        return {'length': np.mean([up - low for up, low in zip(upper_bound, lower_bound)]), 'coverage': 1-out_of_bound/N}
    else:
        lengths = [up - low for up, low in zip(upper_bound, lower_bound)]
        return {'length': np.mean(lengths), 'std_length': np.std(lengths), 'coverage': 1-out_of_bound/N}

def compute_quantile_residual(y, y_hat, r_hat, alpha):
    N = len(y)
    q_yhat = np.quantile(np.abs(np.array(y)-np.array(y_hat))/r_hat,np.ceil((N+1)*(1-alpha))/N)
    return  q_yhat

def eval_regression(ytrue, ypred):
    return {'r2': round(r2_score(ytrue, ypred), 3),
            'rmse' : round(np.sqrt(mean_squared_error(ytrue, ypred)), 3),
            'mae' : round(mean_absolute_error(ytrue, ypred), 3)}

def separate_features_target(df):
    X = [list(item) for item in np.array(df.drop(['fname', 'ytrue', 'ypred', 'target'], axis=1))]
    r, y, y_hat = list(df['target']), list(df['ytrue']), list(df['ypred'])
    return X, r, y, y_hat

def get_fnames(pids, dmmld=True): 
    if dmmld:
        clinical_data = pd.read_csv('data/csv_files/SCR-003.Clinical.Data.csv')
    else:
        clinical_data = pd.read_csv('data/csv_files/SCR-002.Clinical.Data.csv')
    fnames = list(clinical_data[clinical_data.RANDID.isin(pids)].EGREFID)
    return fnames

def split_patients(dmmld=True): 
    if dmmld:
        clinical_data = pd.read_csv('data/csv_files/SCR-003.Clinical.Data.csv')
        pids = list(set(clinical_data.RANDID))
        random.seed(3)
        random.shuffle(pids)
        return pids[:7], pids[7:14], pids[14:]
    else:
        clinical_data = pd.read_csv('data/csv_files/SCR-002.Clinical.Data.csv')
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