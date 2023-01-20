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
parser.add_argument('--split_dmmld', default="False", help="")

def bayes_pred_intervals(qts, alpha = 0.1, hpd = False, variance = False):  
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

def conformal_pred_intervals(alpha, model_r, X_cal2, r_cal2, y_cal2, y_hat_cal2, X_val, r_val, y_val, y_hat_val):   
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

    split_dmmld = eval(args.split_dmmld)
    if split_dmmld:
        pids_cal1, pids_cal2, pids_val = utils.split_patients(dmmld=True) 

    alphas = np.arange(0.05, 1, 0.05)

    if args.method == 'bayes':  
        if not os.path.exists('data/json_files'):
            print('missing json files, request them from authors')
        else:
            json_path, nfold = 'data/json_files', 5 

            for agg_method in ['global', 'leads', 'folds']:
                print(f'{args.method} - {agg_method}')
                dic = {'rdvq': {'alpha': [], 'cvg': [], 'length' : []},
                    'dmmld': {'alpha': [],'cvg': [], 'length' : []}}

                for folder in [ 'rdvq', 'dmmld']:  
                    if folder == 'dmmld' and split_dmmld:  
                        fnames = utils.get_fnames(pids_val)
                        pathlist = [os.path.join(json_path, folder, f'{fname}.json') for fname in fnames]
                    else:
                        pathlist = list(Path(os.path.join(json_path, folder)).glob('**/*.json')) 
                
                    leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] 
        
                    with tqdm(total = len(alphas)) as pbar:
                        for i, alpha in enumerate(alphas): 
                            lower_bound, upper_bound, y_true = [], [], [] 
                            for path in pathlist:
                                sample = json.load(open(path, "r")) 
                                qt_ref = sample['qt_ref']
                                all_preds = [[sample['outputs'][lead][f"qt_{fold+1}"] for fold in range(nfold)] for lead in leads]
                                qt_preds = utils.aggregate_qt_preds(all_preds, method=agg_method) 
                                cis = bayes_pred_intervals(qt_preds, alpha=alpha)  
                                lower_bound.append(cis[0])
                                upper_bound.append(cis[1])
                                y_true.append(qt_ref) 
                            results = utils.calculate_coverage(lower_bound, upper_bound, y_true)    
                            dic[folder]['alpha'].append(alpha) 
                            dic[folder]['length'].append(results['length'])
                            dic[folder]['cvg'].append(results['coverage'])  
                            pbar.update()  
                if split_dmmld:
                    utils.save_dict_to_json(dic, os.path.join('experiments', args.method, 'split_dmmld', f'coverage_{agg_method}.json'))
                else:
                    utils.save_dict_to_json(dic, os.path.join('experiments', args.method, 'full_dmmld', f'coverage_{agg_method}.json'))
    elif args.method == 'conformal': 
        print('loading data')
        df_rdvq = pd.read_csv('data/csv_files/df_rdvq.csv')
        df_dmmld = pd.read_csv('data/csv_files/df_dmmld.csv')

        if not split_dmmld:
            pids_cal1, pids_cal2 = utils.split_patients(dmmld=False) 
            df_rdvq_cal1 = df_rdvq[df_rdvq.fname.isin(utils.get_fnames(pids_cal1, dmmld=False))] 
            df_rdvq_cal2 = df_rdvq[df_rdvq.fname.isin(utils.get_fnames(pids_cal2, dmmld=False))] 
            X_cal1, r_cal1, y_cal1, y_hat_cal1 = utils.separate_features_target(df_rdvq_cal1)
            X_cal2, r_cal2, y_cal2, y_hat_cal2 = utils.separate_features_target(df_rdvq_cal2) 
            X_val, r_val, y_val, y_hat_val = utils.separate_features_target(df_dmmld)  
            dir = os.path.join('experiments', args.method, 'full_dmmld')
        else:
            df_dmmld_cal1 = df_dmmld[df_dmmld.fname.isin(utils.get_fnames(pids_cal1))] 
            df_dmmld_cal2 = df_dmmld[df_dmmld.fname.isin(utils.get_fnames(pids_cal2))] 
            df_dmmld_val = df_dmmld[df_dmmld.fname.isin(utils.get_fnames(pids_val))]  
            X_cal1, r_cal1, y_cal1, y_hat_cal1 = utils.separate_features_target(df_dmmld_cal1)
            X_cal2, r_cal2, y_cal2, y_hat_cal2 = utils.separate_features_target(df_dmmld_cal2)
            X_val, r_val, y_val, y_hat_val = utils.separate_features_target(df_dmmld_val) 
            dir = os.path.join('experiments', args.method, 'split_dmmld')
         
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

            # model = GradientBoostingRegressor()
            # grid = dict()
            # grid['n_estimators'] = [50, 100, 200, 300, 500, 800, 1000]
            # grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
            # grid['subsample'] = [0.5, 0.7, 1.0]
            # grid['max_depth'] = [10]  
            # grid["warm_start"] = [True]
            # grid["max_features"] = [7]

            # # define the grid search procedure
            # grid_search = GridSearchCV(estimator=model, param_grid=grid, verbose=5)
            # # execute the grid search
            # grid_result = grid_search.fit(X_cal1, y_cal1)
            # # summarize the best score and configuration
            # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) # summarize all scores that were evaluated
            # means = grid_result.cv_results_['mean_test_score']
            # stds = grid_result.cv_results_['std_test_score']
            # params = grid_result.cv_results_['params']
            # for mean, stdev, param in zip(means, stds, params):
            #     print("%f (%f) with: %r" % (mean, stdev, param)) 

        dic = {'dmmld': {'alpha': [],'cvg': [], 'length' : []}}
        with tqdm(total = len(alphas)) as pbar:
            for alpha in alphas:
                lower_bound, upper_bound, results =  conformal_pred_intervals(alpha, model_r, X_cal2, r_cal2, y_cal2, y_hat_cal2, X_val, r_val, y_val, y_hat_val)
                dic['dmmld']['alpha'].append(alpha) 
                dic['dmmld']['length'].append(results['length'])
                dic['dmmld']['cvg'].append(results['coverage']) 
                pbar.update()
        utils.save_dict_to_json(dic, os.path.join(dir, 'coverage.json'))  
        
       
       
       
       
       
       
       
       
       
       
       
