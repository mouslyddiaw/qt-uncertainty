import argparse
import json
import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.utils import shuffle 
import random
from pathlib import Path
from tqdm import tqdm
import seaborn as sns
import utils
from get_pred_intervals import bayes_pred_interval, conformal_pred_interval 

parser = argparse.ArgumentParser()
parser.add_argument('--result_type', default='0', help="ranges from 1 to 5, generates the different results in the paper")  

def get_pis_bayes(fnames, nfold=5, nleads=12, agg_method='global', alpha=0.1, study='dmmld'): 
    df = pd.read_csv(f'data/df_{study}.csv')  
    lower_bound , upper_bound, mean_qts = [], [], [] 
    for fname in fnames:
        row = df[df.fname==fname]  
        all_preds = [[row[f'ypred_{k +l*nfold}'] for k in range(1, nfold+1)] for l in range(nleads)] 
        qt_preds = utils.aggregate_qt_preds(all_preds, method=agg_method) 
        cis = bayes_pred_interval(qt_preds, alpha=alpha)  
        lower_bound.append(cis[0])
        upper_bound.append(cis[1])
        mean_qts.append(np.mean(qt_preds))
    return lower_bound, upper_bound, mean_qts

def get_frst_fname_triplicate(data, patient_id, tpt, drug): 
    rows = data[(data.RANDID==patient_id) & (data.TRTA==drug) & (data.TPT==tpt)]
    return rows.EGREFID.iloc[0] 

if __name__ == "__main__":
    args = parser.parse_args()  
    result_type = int(args.result_type)
    if result_type == 1:
        print(' ')
        print('Length and coverage (alpha=0.1)')
        method, study = 'conformal', 'dmmld'
        print(f'-------------- {method}--------------') 
        dic  = json.load(open(os.path.join('experiments', method, 'coverage.json'), "r"))[study] 
        for alpha, cvg, lngth, dev in zip(dic['alpha'], dic['cvg'], dic['length'], dic['deviation']):
            if alpha == 0.1:
                print(f'({study}) Coverage: {round(cvg, 2)}, Length: {round(lngth, 2)}, Deviation: {round(dev, 2)}')

        method = 'bayes' 
        print(f'-------------- {method}--------------')   
        for agg_method in ['global', 'leads']: #global: UQ-ELM, leads: UQ-EL
            dic_full  = json.load(open(os.path.join('experiments', method, f'coverage_{agg_method}.json'), "r"))  
            for study in ['rdvq', 'dmmld']: 
                dic = dic_full[study]
                for alpha, cvg, lngth, dev in zip(dic['alpha'], dic['cvg'], dic['length'], dic['deviation']):
                    if alpha == 0.1:
                        print(f'({study}, {agg_method}) Coverage: {round(cvg, 2)}, Length: {round(lngth, 2)}, Deviation: {round(dev, 2)}')
        print(' ')
    elif result_type == 2: 
        print('Plotting coverage vs alpha')
        dic_conformal  = json.load(open(os.path.join('experiments', 'conformal', 'coverage.json'), "r")) 
        dic_eml  = json.load(open(os.path.join('experiments', 'bayes', 'coverage_global.json'), "r"))  
        dic_el  = json.load(open(os.path.join('experiments', 'bayes', 'coverage_leads.json'), "r")) 
        alphas = dic_eml['dmmld']['alpha']  
        fontsize, markersize = 20, 12
        plt.figure(dpi=400, figsize=(9, 5)) 
        plt.plot(list(alphas), utils.convert_to_percent(dic_el['rdvq']['cvg']), "-*", label='S1b/UQ-EL', color='tab:blue', markersize=markersize) 
        plt.plot(list(alphas), utils.convert_to_percent(dic_eml['rdvq']['cvg']), '-o', label='S1b/UQ-ELM', color='tab:blue', markersize=markersize) 
        plt.plot(list(alphas), utils.convert_to_percent(dic_el['dmmld']['cvg']), "-*", label='S2/UQ-EL', color='tab:orange', markersize=markersize) 
        plt.plot(list(alphas), utils.convert_to_percent(dic_eml['dmmld']['cvg']), '-o', label='S2/UQ-ELM', color='tab:orange', markersize=markersize)
        plt.plot(list(alphas), utils.convert_to_percent(dic_conformal['dmmld']['cvg']), "-^", label='S2/LASCP', color='tab:orange', markersize=markersize) 
        plt.plot(list(alphas), [100*(1-alpha) for alpha in list(alphas)], '--', color= 'k', label = 'y = 100(1-α)%', linewidth=3) 
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('Alpha', fontsize = fontsize)
        plt.ylabel('Coverage (%)', fontsize = fontsize)
        plt.tick_params(labelsize=fontsize)  
        plt.legend(frameon=False, loc ='lower left', fontsize=16.8 ) 
        plt.tight_layout()
        plt.savefig('experiments/imgs/cvg_vs_alpha.png') 
        print(' ')
    elif result_type == 3: 
        print('Evaluating residual fitting')
        df_rdvq = pd.read_csv('data/df_rdvq.csv')
        df_dmmld = pd.read_csv('data/df_dmmld.csv') 
        
        pids_cal1, pids_cal2 = utils.split_patients(dmmld=False) 
        df_rdvq_cal1 = df_rdvq[df_rdvq.fname.isin(utils.get_fnames(pids_cal1, dmmld=False))] 
        df_rdvq_cal2 = df_rdvq[df_rdvq.fname.isin(utils.get_fnames(pids_cal2, dmmld=False))] 
        X_cal1, r_cal1, y_cal1, y_hat_cal1 = utils.separate_features_target(df_rdvq_cal1)
        X_cal2, r_cal2, y_cal2, y_hat_cal2 = utils.separate_features_target(df_rdvq_cal2) 
        X_val, r_val, y_val, y_hat_val = utils.separate_features_target(df_dmmld)  
        dir = os.path.join('experiments','conformal') 

        model_r = utils.load_model(filename = os.path.join(dir, 'model', 'GBReg.sav')) 

        print('Calib 1', utils.eval_regression(r_cal1, model_r.predict(X_cal1)))
        print('Calib 2', utils.eval_regression(r_cal2, model_r.predict(X_cal2)))
        print('Validation', utils.eval_regression(r_val, model_r.predict(X_val))) 
        print(' ')
    elif result_type == 4:   
        print('Plotting 24-h QT profiles')
        clinical_data = pd.read_csv('data/SCR-003.Clinical.Data.csv')
        df_rdvq = pd.read_csv('data/df_rdvq.csv')
        df_dmmld = pd.read_csv('data/df_dmmld.csv')

        pids_cal1, pids_cal2 = utils.split_patients(dmmld=False) 
        df_rdvq_cal1 = df_rdvq[df_rdvq.fname.isin(utils.get_fnames(pids_cal1, dmmld=False))] 
        df_rdvq_cal2 = df_rdvq[df_rdvq.fname.isin(utils.get_fnames(pids_cal2, dmmld=False))] 
        X_cal1, r_cal1, y_cal1, y_hat_cal1 = utils.separate_features_target(df_rdvq_cal1)
        X_cal2, r_cal2, y_cal2, y_hat_cal2 = utils.separate_features_target(df_rdvq_cal2) 
        X_val, r_val, y_val, y_hat_val = utils.separate_features_target(df_dmmld)  
        dir = os.path.join('experiments','conformal') 
        model_r = utils.load_model(filename = os.path.join(dir, 'model', 'GBReg.sav'))

        fig, ax = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(16, 6))
        for compt1, idx in enumerate([np.argmin(y_val), np.argmax(y_val)]):
            fname = df_dmmld.fname.iloc[idx]
            
            pid = clinical_data[clinical_data.EGREFID == fname].RANDID.iloc[0] 
            drug = clinical_data[clinical_data.EGREFID == fname].TRTA.iloc[0]
            print(pid, drug) 
            timepoints = list(set(clinical_data.TPT))
            fnames = [get_frst_fname_triplicate(clinical_data, pid, tpt, drug) for tpt in timepoints] 

            alpha = 0.1
            qts = [clinical_data[clinical_data.EGREFID == fname].QT.iloc[0] for fname in fnames]
            lower_bayes, upper_bayes, qts_pred = get_pis_bayes(fnames, alpha=alpha, agg_method='leads') 
            lower_conformal, upper_conformal, _ = conformal_pred_interval(alpha, model_r, X_cal2, r_cal2, y_cal2, y_hat_cal2, X_val, r_val, y_val, y_hat_val)
            idx_fnames = [list(df_dmmld[df_dmmld.fname == fname].index)[0] for fname in fnames] 
            lower_conformal, upper_conformal = lower_conformal[idx_fnames], upper_conformal[idx_fnames]

            for compt2, (method, color_fill, color_line) in enumerate(zip(['UQ-EL', 'LASCP'], ['#edf2fb', '#f5f5f5'], ['#4895ef','#482e77'])): 
                if method == 'UQ-EL':
                    lower, upper = lower_bayes, upper_bayes
                else:
                    lower, upper = lower_conformal, upper_conformal 
                fontsize = 19
                ax[compt1][compt2].fill_between(timepoints, lower, upper, color=color_fill, label=method)   
                ax[compt1][compt2].plot(timepoints, lower, color=color_line)
                ax[compt1][compt2].plot(timepoints, upper, color=color_line) 
                ax[compt1][compt2].plot(timepoints, qts, '-o', markersize=10,color='darkgray', label = 'Ground truth',linewidth=0.5)
                ax[compt1][compt2].tick_params(labelsize=fontsize)
                if compt1 == 1:
                    ax[compt1][compt2].set_xlabel('Time (h)', fontsize=fontsize)
                if compt2 == 0:
                    ax[compt1][compt2].set_ylabel('QT (ms)', fontsize=fontsize)   
        plt.tight_layout()
        plt.savefig(f'experiments/imgs/qt_profile.png') 
        print('')
    elif result_type == 5:
        print('Error-based calibration plot')
        alpha, samples_per_bin, study = 0.1, 100 , 'dmmld' 
        df_rdvq = pd.read_csv('data/df_rdvq.csv') 
        df = pd.read_csv(f'data/df_{study}.csv')  
        _, pids_cal2 = utils.split_patients(dmmld=False) 
        df_rdvq_cal2 = df_rdvq[df_rdvq.fname.isin(utils.get_fnames(pids_cal2, dmmld=False))] 
        X_cal2, r_cal2, y_cal2, y_hat_cal2 = utils.separate_features_target(df_rdvq_cal2) 
        X_val, r_val, y_val, y_hat_val = utils.separate_features_target(df)
        dir = os.path.join('experiments', 'conformal') 
        
        fnames = list(df.fname) 
        ytrue = df.ytrue
        model_r = utils.load_model(filename = os.path.join(dir, 'model', 'GBReg.sav'))
        lower_conformal, upper_conformal, ypred = conformal_pred_interval(alpha, model_r, X_cal2, r_cal2, y_cal2, y_hat_cal2, X_val, r_val, y_val, y_hat_val)
        lower_bayes_glob, upper_bayes_glob, ypred = get_pis_bayes(fnames, nfold=5, agg_method='global', alpha=alpha, study=study) 
        lower_bayes, upper_bayes, ypred = get_pis_bayes(fnames, nfold=5, agg_method='leads', alpha=alpha, study=study) 

        _, widths_bin_bayes_glob = utils.binned_err(ytrue, ypred, lower_bayes_glob, upper_bayes_glob, samples_per_bin=samples_per_bin)
        _, widths_bin_bayes = utils.binned_err(ytrue, ypred, lower_bayes, upper_bayes, samples_per_bin=samples_per_bin)
        errors_bin, widths_bin_conformal = utils.binned_err(ytrue, ypred, lower_conformal, upper_conformal, samples_per_bin=samples_per_bin)
        fontsize = 40
        plt.figure(figsize = (10, 8), dpi=400) 
        plt.plot(errors_bin, widths_bin_bayes, '-o', label='UQ-EL', linewidth=4, color='#4895ef')  
        plt.plot(errors_bin, widths_bin_bayes_glob,'-o', label='UQ-ELM', linewidth=4, color = '#34a0a4')
        plt.plot(errors_bin, widths_bin_conformal,'-o', label='LASCP', linewidth=4, color='#482e77') 
        plt.ylabel('MW (ms)', fontsize=fontsize)
        plt.xlabel('MAE (ms)', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.title(f'S2, α = {alpha}', fontsize=fontsize) 
        plt.legend(frameon=False, fontsize=fontsize-5, loc='upper left')
        plt.tight_layout()
        plt.savefig(f'experiments/imgs/err_vs_pi_width.png')