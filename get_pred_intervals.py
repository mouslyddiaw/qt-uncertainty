import os, json, argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd 
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingRegressor 
from statistics import NormalDist
import pickle 
import utils 

parser = argparse.ArgumentParser()
parser.add_argument('--method', default='bayes', help="bayes or conformal") 

def bayes_pred_interval(qts, alpha = 0.1, hpd = False, variance = False):  
    if hpd: 
        '''Adapted from http://bebi103.caltech.edu.s3-website-us-east-1.amazonaws.com/2015/tutorials/l06_credible_regions.html''' 
        d = np.sort(np.copy(qts)) 
        n = len(qts) 
        mass_frac = 1- alpha 
        n_samples = np.floor(mass_frac * n).astype(int) 
        int_width = d[n_samples:] - d[:n-n_samples] 
        min_int = np.argmin(int_width) 
        return np.array([d[min_int], d[min_int+n_samples]])
    else:
        if not variance: 
            return  np.quantile(qts, [alpha/2, 1-(alpha/2)])    
        else:
            '''cf. Pearce et al, High-Quality Prediction Intervals for Deep Learning (https://arxiv.org/pdf/1802.07167.pdf)'''
            z = NormalDist().inv_cdf((1 + 1 - alpha) / 2.)  
            mean_qt = np.mean(qts)
            sigma2 = sum([(val - mean_qt)**2 for val in qts])/(len(qts)-1) #variance
            sigma = np.sqrt(sigma2)
            return  [np.mean(qts) - z*sigma, np.mean(qts) + z*sigma]

def conformal_pred_interval(alpha, model_r, X_cal2, r_cal2, y_cal2, y_hat_cal2, X_val, r_val, y_val, y_hat_val):   
    '''The notebook available at https://github.com/Quilograma/ConformalPredictionTutorial is helpful to get started with conformal prediction'''
    # calculate q_yhat 
    r_hat_cal2 = model_r.predict(X_cal2) 
    q_yhat = utils.compute_quantile_residual(y_cal2, y_hat_cal2, r_hat_cal2, alpha)  
     
    # predict with 1-alpha confidence  
    r_hat_val = model_r.predict(X_val) 
    lower_bound = y_hat_val - q_yhat*r_hat_val
    upper_bound = y_hat_val + q_yhat*r_hat_val 
    results = utils.calculate_coverage(lower_bound, upper_bound, y_val)
    return lower_bound, upper_bound, results 

if __name__ == '__main__':  

    args = parser.parse_args()  

    alphas = np.arange(0.05, 1, 0.05) 

    if args.method == 'bayes':  
        nfold, nleads = 5, 12 
        for agg_method in ['global', 'leads']:  #global: UQ-ELM, leads: UQ-EL
            print(f'{args.method} - {agg_method}')
            dic = {'rdvq': {'alpha': [], 'cvg': [], 'length' : [], 'deviation': []},
                   'dmmld': {'alpha': [],'cvg': [], 'length' : [], 'deviation': []} }

            for study in dic.keys():  
                df = pd.read_csv(f'data/df_{study}.csv')  
                with tqdm(total = len(alphas)) as pbar:
                    for i, alpha in enumerate(alphas): 
                        lower_bound, upper_bound, y_true = [], [], [] 
                        for _, row in df.iterrows():  
                            qt_ref = row['ytrue']
                            all_preds = [[row[f'ypred_{k +l*nfold}'] for k in range(1, nfold+1)] for l in range(nleads)] 
                            qt_preds = utils.aggregate_qt_preds(all_preds, method=agg_method) 
                            cis = bayes_pred_interval(qt_preds, alpha=alpha)  
                            lower_bound.append(cis[0])
                            upper_bound.append(cis[1])
                            y_true.append(qt_ref) 
                        results = utils.calculate_coverage(lower_bound, upper_bound, y_true)  
                        dic[study]['alpha'].append(alpha) 
                        for key in results.keys():   
                            dic[study][key].append(results[key]) 
                        pbar.update()  
            utils.save_dict_to_json(dic, os.path.join('experiments', args.method, f'coverage_{agg_method}.json'))
    elif args.method == 'conformal': 
        print('conformal')
        df_rdvq = pd.read_csv('data/df_rdvq.csv') 
        pids_cal1, pids_cal2 = utils.split_patients(dmmld=False) 
        df_rdvq_cal1 = df_rdvq[df_rdvq.fname.isin(utils.get_fnames(pids_cal1, dmmld=False))] 
        df_rdvq_cal2 = df_rdvq[df_rdvq.fname.isin(utils.get_fnames(pids_cal2, dmmld=False))] 
        X_cal1, r_cal1, y_cal1, y_hat_cal1 = utils.separate_features_target(df_rdvq_cal1)
        X_cal2, r_cal2, y_cal2, y_hat_cal2 = utils.separate_features_target(df_rdvq_cal2)  
        dir = os.path.join('experiments', args.method) 
         
        try:
            print('loading trained model...') 
            model_r = utils.load_model(filename = os.path.join(dir, 'model', 'GBReg.sav'))
        except FileNotFoundError:
            print('not found, training model') 
            params = json.load(open(os.path.join(dir, 'model', 'params.json'), "r")) 
            params["warm_start"] = eval(params["warm_start"])
            model_r = GradientBoostingRegressor(**params)  
            model_r.fit(X_cal1, r_cal1) 
            pickle.dump(model_r, open(os.path.join(dir, 'model', 'GBReg.sav'), 'wb'))  

        dic = {'dmmld': {'alpha': [],'cvg': [], 'length' : [], 'deviation': []} }

        for study in dic.keys():
            df = pd.read_csv(f'data/df_{study}.csv') 
            X_val, r_val, y_val, y_hat_val = utils.separate_features_target(df)  
            with tqdm(total = len(alphas)) as pbar:
                for alpha in alphas:
                    lower_bound, upper_bound, results =  conformal_pred_interval(alpha, model_r, X_cal2, r_cal2, y_cal2, y_hat_cal2, X_val, r_val, y_val, y_hat_val)
                    dic[study]['alpha'].append(alpha) 
                    for key in results.keys():   
                        dic[study][key].append(results[key])
                    pbar.update()
        utils.save_dict_to_json(dic, os.path.join(dir, 'coverage.json'))  
        
       
       
       
       
       
       
       
       
       
       
       
