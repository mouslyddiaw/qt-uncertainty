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
from get_confidence_intervals import bayes_conf_intervals, conformal_conf_intervals 

parser = argparse.ArgumentParser()
parser.add_argument('--result_type', default='0', help="ranges from 1 to 8, generates the different results in the paper") 

def get_cis_conformal(alpha, model_r, df_dmmld, fnames):
    df_dmmld_cal1 = df_dmmld[df_dmmld.fname.isin(fnames)] 
    df_dmmld_cal2 = df_dmmld[df_dmmld.fname.isin(fnames)] 
    df_dmmld_val = df_dmmld[df_dmmld.fname.isin(fnames)]  
    X_cal1, r_cal1, y_cal1, y_hat_cal1 = utils.separate_features_target(df_dmmld_cal1)
    X_cal2, r_cal2, y_cal2, y_hat_cal2 = utils.separate_features_target(df_dmmld_cal2)
    X_val, r_val, y_val, y_hat_val = utils.separate_features_target(df_dmmld_val) 
    lower_conformal, upper_conformal, _ = conformal_conf_intervals(alpha, model_r, X_cal2, r_cal2, y_cal2, y_hat_cal2, X_val, r_val, y_val, y_hat_val)
    return lower_conformal, upper_conformal

def get_cis_bayes(fnames, nfold=5, agg_method='global', alpha=0.1):
    leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] 
    lower_bound , upper_bound, mean_qts = [], [], []
    for fname in fnames:
        sample = json.load(open(f'data/json_files/dmmld/{fname}.json', "r"))  
        all_preds = [[sample['outputs'][lead][f"qt_{fold+1}"] for fold in range(nfold)] for lead in leads]
        qt_preds = utils.aggregate_qt_preds(all_preds, method=agg_method) 
        cis = bayes_conf_intervals(qt_preds, alpha=alpha)  
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
        method = 'conformal' 
        print(f'-------------- {method}--------------')
        for folder in ['full_dmmld', 'split_dmmld']:
            dic  = json.load(open(os.path.join('experiments', method, folder, 'coverage.json'), "r"))['dmmld'] 
            for alpha, cvg, lngth in zip(dic['alpha'], dic['cvg'], dic['length']):
                if alpha == 0.1:
                    print(f'({folder}) Coverage: {round(cvg, 2)}, Length: {round(lngth, 2)}')

        method = 'bayes' 
        print(f'-------------- {method}--------------') 
        for folder in ['full_dmmld', 'split_dmmld']:
            for db in ['rdvq', 'dmmld']:
                if folder == 'split_dmmld' and db=='rdvq':
                    continue
                for agg_method in ['global', 'leads', 'folds']:
                    dic_all_db  = json.load(open(os.path.join('experiments', method, folder, f'coverage_{agg_method}.json'), "r"))  
                    dic = dic_all_db[db]
                    for alpha, cvg, lngth in zip(dic['alpha'], dic['cvg'], dic['length']):
                        if alpha == 0.1:
                            print(f'({folder}, {db}, {agg_method}) Coverage: {round(cvg, 2)}, Length: {round(lngth, 2)}')
                print(' ')
    elif result_type == 2: 
        print('Plotting coverage vs alpha')
        dic_conformal_full  = json.load(open(os.path.join('experiments', 'conformal', 'full_dmmld', 'coverage.json'), "r")) 
        dic_bayes_full  = json.load(open(os.path.join('experiments', 'bayes', 'full_dmmld', 'coverage_global.json'), "r")) 
        dic_conformal_split  = json.load(open(os.path.join('experiments', 'conformal', 'split_dmmld', 'coverage.json'), "r"))
        dic_bayes_split  = json.load(open(os.path.join('experiments', 'bayes', 'split_dmmld', 'coverage_global.json'), "r"))  

        alphas = dic_bayes_full['dmmld']['alpha'] 
        
        fontsize = 15
        fig, ax = plt.subplots(ncols = 2, figsize= (8, 4) , dpi = 400) 
        ax[0].plot(list(alphas), utils.convert_to_percent(dic_bayes_full['rdvq']['cvg']), '-o', label='UQ-EML/DRP', color='tab:blue') 
        ax[0].plot(list(alphas), utils.convert_to_percent(dic_bayes_full['dmmld']['cvg']), '-*', label='UQ-EML/DMMLD', color='tab:blue') 
        ax[0].plot(list(alphas), utils.convert_to_percent(dic_conformal_full['dmmld']['cvg']), '-*', label='LASCP/DMMLD', color='tab:orange')
        ax[0].plot(list(alphas), [100*(1-alpha) for alpha in list(alphas)], '--', color= 'k')
        ax[0].set_title('Experiment 1', fontsize = fontsize)
        ax[0].set_xlabel('Alpha', fontsize = fontsize)
        ax[0].set_ylabel('Coverage (%)', fontsize = fontsize)
        ax[0].tick_params(labelsize=fontsize) 
        ax[0].legend(frameon=False, loc ='lower left')

        ax[1].plot(list(alphas), utils.convert_to_percent(dic_bayes_split['dmmld']['cvg']), '-*', label='UQ-EML/DMMLD', color='tab:blue') 
        ax[1].plot(list(alphas), utils.convert_to_percent(dic_conformal_split['dmmld']['cvg']), '-*', label='LASCP/DMMLD', color='tab:orange')  
        ax[1].plot(list(alphas), [100*(1-alpha) for alpha in list(alphas)], '--', color= 'k')
        ax[1].set_title('Experiment 2', fontsize = fontsize)
        ax[1].set_xlabel('Alpha', fontsize = fontsize)
        ax[1].tick_params(labelsize=fontsize) 
        ax[1].legend(frameon=False, loc ='lower left')
        # plt.legend()
        plt.tight_layout()
        plt.savefig('experiments/imgs/cvg_vs_alpha.png') 
    elif result_type == 3: 
        print('Evaluate residual fitting')
        df_rdvq = pd.read_csv('data/csv_files/df_rdvq.csv')
        df_dmmld = pd.read_csv('data/csv_files/df_dmmld.csv')

        for split_dmmld in [False, True]:
            if not split_dmmld:
                print('-- full_dmmld')
                df_rdvq = shuffle(df_rdvq)  
                X_cal1, r_cal1, y_cal1, y_hat_cal1 = utils.separate_features_target(df_rdvq.iloc[:70*len(df_rdvq)//100,:])
                X_cal2, r_cal2, y_cal2, y_hat_cal2 = utils.separate_features_target(df_rdvq.iloc[70*len(df_rdvq)//100:,:]) 
                X_val, r_val, y_val, y_hat_val = utils.separate_features_target(df_dmmld)  
                dir = os.path.join('experiments', 'conformal', 'full_dmmld')
            else:
                print('-- split_dmmld')
                pids_cal1, pids_cal2, pids_val = utils.split_patients(dmmld=True) 
                df_dmmld_cal1 = df_dmmld[df_dmmld.fname.isin(utils.get_fnames(pids_cal1))] 
                df_dmmld_cal2 = df_dmmld[df_dmmld.fname.isin(utils.get_fnames(pids_cal2))] 
                df_dmmld_val = df_dmmld[df_dmmld.fname.isin(utils.get_fnames(pids_val))]  
                X_cal1, r_cal1, y_cal1, y_hat_cal1 = utils.separate_features_target(df_dmmld_cal1)
                X_cal2, r_cal2, y_cal2, y_hat_cal2 = utils.separate_features_target(df_dmmld_cal2)
                X_val, r_val, y_val, y_hat_val = utils.separate_features_target(df_dmmld_val) 
                dir = os.path.join('experiments', 'conformal', 'split_dmmld')

            model_r = utils.load_model(filename = os.path.join(dir, 'model', 'GBReg.sav')) 

            print('Calib 1', utils.eval_regression(r_cal1, model_r.predict(X_cal1)))
            print('Calib 2', utils.eval_regression(r_cal2, model_r.predict(X_cal2)))
            print('Validation', utils.eval_regression(r_val, model_r.predict(X_val)))
            print(' ')
    elif result_type == 4:  
        if not os.path.exists('data/json_files'):
            print('missing json files, request them from authors')
        else:
            clinical_data = pd.read_csv('data/csv_files/SCR-003.Clinical.Data.csv')
            df_rdvq = pd.read_csv('data/csv_files/df_rdvq.csv')
            df_dmmld = pd.read_csv('data/csv_files/df_dmmld.csv')
            dir = os.path.join('experiments', 'conformal', 'split_dmmld')
            model_r = utils.load_model(filename = os.path.join(dir, 'model', 'GBReg.sav'))
    
            pids_cal1, pids_cal2, pids_val = utils.split_patients(dmmld=True) 
            df_dmmld_cal1 = df_dmmld[df_dmmld.fname.isin(utils.get_fnames(pids_cal1))] 
            df_dmmld_cal2 = df_dmmld[df_dmmld.fname.isin(utils.get_fnames(pids_cal2))] 
            df_dmmld_val = df_dmmld[df_dmmld.fname.isin(utils.get_fnames(pids_val))]  
            X_cal1, r_cal1, y_cal1, y_hat_cal1 = utils.separate_features_target(df_dmmld_cal1)
            X_cal2, r_cal2, y_cal2, y_hat_cal2 = utils.separate_features_target(df_dmmld_cal2)
            X_val, r_val, y_val, y_hat_val = utils.separate_features_target(df_dmmld_val) 

            for compt, idx in enumerate([np.argmin(y_val), np.argmax(y_val)]):
                fname = df_dmmld_val.fname.iloc[idx]
                
                pid = clinical_data[clinical_data.EGREFID == fname].RANDID.iloc[0]
                drug = clinical_data[clinical_data.EGREFID == fname].TRTA.iloc[0]
                # print(pid, drug)

                timepoints = list(set(clinical_data.TPT))
                fnames = [get_frst_fname_triplicate(clinical_data, pid, tpt, drug) for tpt in timepoints] 

                alpha = 0.1
                lower_bayes, upper_bayes, qts_pred = get_cis_bayes(fnames, alpha=alpha) 
                lower_conformal, upper_conformal = get_cis_conformal(alpha, model_r, df_dmmld, fnames)
        
                qts = [clinical_data[clinical_data.EGREFID == fname].QT.iloc[0] for fname in fnames] 
                fontsize = 15
                plt.figure(dpi = 400, figsize = (8, 4))
                plt.plot(timepoints, qts, '-*', color='k', label = 'Actual QT')
                plt.plot(timepoints, qts_pred, '--', color='k', label = 'Mean pred. QT')
                plt.plot(timepoints, lower_conformal, '--', color='tab:purple', label = 'LASCP', linewidth=0.5)
                plt.plot(timepoints, upper_conformal, '--', color='tab:purple', linewidth=0.5) 
                plt.fill_between(timepoints, lower_bayes, upper_bayes, color='silver', label = 'UQ-EML') 
                plt.ylim(350, 500) 
                plt.xlabel('Time (h)', fontsize=fontsize)
                plt.ylabel('QT (ms)', fontsize=fontsize)
                plt.title(f'id:{pid} on {drug}', fontsize=fontsize)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                if compt == 0:
                    plt.legend(frameon = False, fontsize=fontsize)
                plt.tight_layout()
                plt.savefig(f'experiments/imgs/qt_profile_{compt+1}.png')  

                if compt == 1: 
                    plt.figure(dpi=400)
                    for idx_ax, (idx_ecg_illustr, color_ecg, color_pi) in enumerate(zip([0, 9], ['tab:blue', 'tab:purple'], ['tab:blue', 'silver'])):
                        leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] 
                        fname, tpt = fnames[idx_ecg_illustr], timepoints[idx_ecg_illustr]
                        qt, qt_pred = qts[idx_ecg_illustr], qts_pred[idx_ecg_illustr]
                        low, up = lower_bayes[idx_ecg_illustr], upper_bayes[idx_ecg_illustr]
                        qon_labels = pd.read_csv('../data/TQTstudy2/qon_labels.csv') 
                        qon = list(qon_labels[qon_labels['EGREFID']==fname]['QON'])[0] 
                        dic = json.load(open(f'../data/TQTstudy2/custom_templates/{fname}.json', "r"))  
                        templates = [[float(val) for val in lst] for lst in dic['templates']]
                        templates.reverse()
                        leads.reverse()
                        time = [2*i for i in range(600)] 
                        for idx, (lead, template) in enumerate(zip(leads, templates)):
                            plt.plot(time, [val+0.25*(idx+1) for val in template], color = color_ecg) 
                        if idx_ax == 1:
                            plt.axvline(qon, linewidth=0.5, color=color_ecg)
                            plt.axvline(qon+qt, color = 'k') 
                            plt.axvline(qon+qt_pred, linestyle='--', color = 'k') 
                            plt.fill_betweenx(np.arange(0, 4.5), qon+low, qon+up, color=color_pi)
                            plt.yticks([0.25*(idx+1) for idx in range(0, 12)], leads)
                        else:
                            plt.axvline(qon+low, color = color_pi) 
                            plt.axvline(qon+up, color = color_pi) 
                            plt.axvline(qon+qt_pred, linestyle='--', color = color_pi)  
                    plt.ylim(0, 3.65) 
                    for pos in ['right', 'top']:
                        plt.gca().spines[pos].set_visible(False)
                    plt.xlabel('Time (ms)')
                    plt.tight_layout()
                    plt.savefig(f'experiments/imgs/ecg_pi_illustr.png')
    elif result_type == 5:  
        if not os.path.exists('data/json_files'):
            print('missing json files, request them from authors')
        else:
            leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] 
            clinical_data = pd.read_csv('data/csv_files/SCR-003.Clinical.Data.csv') 
            fnames = list(clinical_data.EGREFID)
            random.seed(2)
            random.shuffle(fnames) 
            fname = fnames[0] 

            nfold, agg_method, alpha = 5, 'global', 0.1
            sample = json.load(open(f'data/json_files/dmmld/{fname}.json', "r"))  
            qt_ref = sample['qt_ref']
            all_preds = [[sample['outputs'][lead][f"qt_{fold+1}"] for fold in range(nfold)] for lead in leads]
            qt_preds = utils.aggregate_qt_preds(all_preds, method=agg_method) 
            qt_pred = np.mean(qt_preds)
            cis = bayes_conf_intervals(qt_preds, alpha=alpha)   

            k = -0.001
            plt.figure(dpi=400, figsize=(12,1))
            plt.axhline(k, linestyle='--', color='silver', linewidth=0.5) 
            for x in qt_preds:
                plt.plot(x, k, 'o', color='k')
            plt.plot(qt_ref, k, 'X', color='green', markersize = 10)
            plt.plot(qt_pred, k, 'X', color='blue', markersize = 10)
            plt.plot(cis[0], k, '|', color='k', markersize = 40)
            plt.plot(cis[1], k, '|', color='k', markersize = 40)
            eps=0.1
            plt.ylim(-0.02, 0.055)
            # plt.ylim(1-eps, 1+eps) 
            # plt.xlim(350, 405)  
            for pos in ['right', 'top', 'left']:
                plt.gca().spines[pos].set_visible(False)
            sns.kdeplot(qt_preds, bw_adjust=.35)
            plt.yticks([])
            plt.xticks(np.arange(330, 500, 10), [f'{int(val)}ms' for val in np.arange(330, 500,10)], fontsize=10)
            plt.tight_layout()
            plt.savefig(f'experiments/imgs/illustr_mf.png')

            dic = json.load(open(f'../data/TQTstudy2/custom_templates/{fname}.json', "r"))  
            templates = [[float(val) for val in lst] for lst in dic['templates']]

            for lead, template in zip(leads, templates):
                if lead not in ['I', 'II', 'V6']:
                    continue
                plt.figure(dpi=400, figsize=(12,3))
                plt.plot(template, linewidth=5)
                plt.ylim(-0.4, 1.6)
                plt.xlim(20, 550) 
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f'experiments/imgs/beat_{lead}.png')
    elif result_type == 6:
        df_rdvq = pd.read_csv('data/csv_files/df_rdvq.csv')
        df_dmmld = pd.read_csv('data/csv_files/df_dmmld.csv') 
        for compt, split_dmmld in enumerate([False, True]):
            print(f'---Experiment {compt+1}---')
            if not split_dmmld:
                pids_cal1, pids_cal2 = utils.split_patients(dmmld=False)  
                df_rdvq_cal1 = df_rdvq[df_rdvq.fname.isin(utils.get_fnames(pids_cal1, dmmld=False))] 
                df_rdvq_cal2 = df_rdvq[df_rdvq.fname.isin(utils.get_fnames(pids_cal2, dmmld=False))]  
                print(f'Nb patients - I1 ({len(pids_cal1)}) - I2 ({len(pids_cal2)})')
                print(f'Nb ECGs - I1 ({len(df_rdvq_cal1)}) - I2 ({len(df_rdvq_cal2)})')
            else:
                pids_cal1, pids_cal2, pids_val = utils.split_patients(dmmld=True) 
                df_dmmld_cal1 = df_dmmld[df_dmmld.fname.isin(utils.get_fnames(pids_cal1))] 
                df_dmmld_cal2 = df_dmmld[df_dmmld.fname.isin(utils.get_fnames(pids_cal2))] 
                df_dmmld_val = df_dmmld[df_dmmld.fname.isin(utils.get_fnames(pids_val))]   
                print(f'Nb patients - I1 ({len(pids_cal1)}) - I2 ({len(pids_cal2)}) - Dval ({len(pids_val)})')
                print(f'Nb ECGs - I1 ({len(df_dmmld_cal1)}) - I2 ({len(df_dmmld_cal2)}) - Dval ({len(df_dmmld_val)})')
            print('')
    elif result_type == 7:
        if not os.path.exists('data/json_files'):
            print('missing json files, request them from authors')
        else:
            json_path, nfold = 'data/json_files', 5  
            agg_method = 'global'  
            alphas =  [0.01, 0.05, 0.1, 0.15]  
            methods = ['ETI', 'HPDI', 'Gaussian']
            dic = {'cvg': {alpha: {method: None for method in methods} for alpha in alphas},
                'lengths': {alpha: {method: [] for method in methods} for alpha in alphas}
                    }
            fontsize = 15
            fig, ax = plt.subplots(nrows=2, dpi=400, figsize = (12, 8)) 
            for compt, folder in enumerate([ 'rdvq', 'dmmld']):   
                print(f'-----{folder}-----')
                pathlist = list(Path(os.path.join(json_path, folder)).glob('**/*.json'))  
                leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']  
                with tqdm(total = len(alphas)) as pbar:
                    for i, alpha in enumerate(alphas): 
                        lower_bound = {method: [] for method in methods}
                        upper_bound = {method: [] for method in methods}
                        y_true = []  
                        for path in pathlist:
                            sample = json.load(open(path, "r")) 
                            qt_ref = sample['qt_ref']
                            all_preds = [[sample['outputs'][lead][f"qt_{fold+1}"] for fold in range(nfold)] for lead in leads]
                            qt_preds = utils.aggregate_qt_preds(all_preds, method=agg_method) 
                            for method in methods:
                                if 'ETI' in method:
                                    variance, hpd = False, False
                                    y_true.append(qt_ref)
                                elif 'HPDI' in method:
                                    variance, hpd = False, True
                                else:
                                    variance, hpd = True, False
                                cis  = bayes_conf_intervals(qt_preds, alpha=alpha, variance = variance, hpd = hpd)   
                                lower_bound[method].append(cis[0])
                                upper_bound[method].append(cis[1])  

                        for method in methods:
                            lengths = [up - low for up, low in zip(upper_bound[method], lower_bound[method])]
                            results = utils.calculate_coverage(lower_bound[method], upper_bound[method], y_true, std_length=True) 
                            dic['cvg'][alpha][method] = results['coverage']   
                            dic['lengths'][alpha][method].extend(lengths)  
                        pbar.update()
                
                all_lengths, labels = [], []
                for alpha in alphas:
                    for method in methods:
                        all_lengths.append(dic['lengths'][alpha][method])   
                        cvg = str(round(100*dic['cvg'][alpha][method], 1))  
                        if 'ETI' in method:
                            labels.append(f"{method}\n{cvg}%\n(α = {alpha})")
                        else:
                            labels.append(f"{method}\n{cvg}%")
                
                ax[compt].boxplot(all_lengths, showfliers = False, widths = 0.8)
                if 'dmmld' in folder:
                    ax[compt].set_title('DMMLD', fontsize = fontsize)
                else:
                    ax[compt].set_title('DRP', fontsize = fontsize)
                ax[compt].set_xticks(range(1, len(labels)+1), labels)
                ax[compt].set_ylabel('PI length (ms)', fontsize = fontsize) 
                ax[compt].tick_params(axis='both', labelsize=fontsize)
                ax[compt].set_yticks(np.arange(0, 250, 50))
            plt.tight_layout()
            plt.savefig(f'experiments/imgs/bp_lengths.png')  
    elif result_type == 8:
        if not os.path.exists('data/json_files'):
            print('missing json files, request them from authors')
        else:
            alphas = [0.01] 
            alphas.extend(list(np.arange(0.1, 1, 0.1)))
    
            json_path, nfold = 'data/json_files', 5 
            agg_method = 'global'

            dic = {'ETI':{'rdvq': {'alpha': [], 'cvg': [], 'length' : []},
                'dmmld': {'alpha': [],'cvg': [], 'length' : []}},
                'HPD': {'rdvq': {'alpha': [], 'cvg': [], 'length' : []},
                'dmmld': {'alpha': [],'cvg': [], 'length' : []}}
            }

            for hpd, tag in zip([False, True], ['ETI', 'HPD']):
                for folder in [ 'rdvq', 'dmmld']:  
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
                                cis = bayes_conf_intervals(qt_preds, alpha=alpha, hpd=hpd)  
                                lower_bound.append(cis[0])
                                upper_bound.append(cis[1])
                                y_true.append(qt_ref) 
                            results = utils.calculate_coverage(lower_bound, upper_bound, y_true)   
                            dic[tag][folder]['alpha'].append(alpha) 
                            dic[tag][folder]['length'].append(results['length']) 
                            dic[tag][folder]['cvg'].append(results['coverage'])  
                            pbar.update() 

            fontsize = 15
            fig, ax = plt.subplots(ncols = 2, figsize= (10, 4) , dpi = 400)
            ax[0].plot(list(alphas), utils.convert_to_percent(dic['HPD']['rdvq']['cvg']), '-o', label='HPDI/DRP', color='tab:orange')
            ax[0].plot(list(alphas), utils.convert_to_percent(dic['HPD']['dmmld']['cvg']), '-*', label='HPDI/DMMLD', color='tab:orange')  
            ax[0].plot(list(alphas), utils.convert_to_percent(dic['ETI']['rdvq']['cvg']), '-o', label='ETI/DRP', color='tab:blue') 
            ax[0].plot(list(alphas), utils.convert_to_percent(dic['ETI']['dmmld']['cvg']), '-*', label='ETI/DMMLD', color='tab:blue') 
            ax[0].plot(list(alphas), [100*(1-alpha) for alpha in list(alphas)], '--', color= 'k') 
            ax[0].set_xticks(np.arange(0, 1, 0.1))
            ax[0].set_xlabel('Alpha', fontsize = fontsize)
            ax[0].set_ylabel('Coverage (%)', fontsize = fontsize)
            ax[0].tick_params(labelsize=fontsize) 
            ax[0].legend(frameon=False, loc ='lower left')
            
            ax[1].plot(list(alphas), dic['HPD']['rdvq']['length'], '-o', label='HPDI/DRP', color='tab:orange')
            ax[1].plot(list(alphas), dic['HPD']['dmmld']['length'], '-*', label='HPDI/DMMLD', color='tab:orange') 
            ax[1].plot(list(alphas), dic['ETI']['rdvq']['length'], '-o', label='ETI/DRP', color='tab:blue') 
            ax[1].plot(list(alphas), dic['ETI']['dmmld']['length'], '-*', label='ETI/DMMLD', color='tab:blue')  
            ax[1].set_xticks(np.arange(0, 1, 0.1))
            ax[1].set_xlabel('Alpha', fontsize = fontsize)
            ax[1].set_ylabel('Average PI length (ms)', fontsize = fontsize)
            ax[1].tick_params(labelsize=fontsize)   
            plt.tight_layout()
            plt.savefig('experiments/imgs/eti_vs_hpd.png') 
            
            
            
            


        

                
    
        
          
