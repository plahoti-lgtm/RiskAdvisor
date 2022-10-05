import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, average_precision_score, brier_score_loss, balanced_accuracy_score, precision_score
from utils.uncertainty import get_aleatoric_epistemic_uncertainities, computeTrustScore, computeConfidence
import numpy as np
from scipy.stats import rankdata

def normalize(arr, norm = 'l2'):
    ''' Args:
            np.array
        returns:
            array after sum to 1 normalization'''
    if norm =='l2':
        return arr/np.sum(arr)
    elif norm=='min-max':
        return (arr - np.min(arr))/(np.max(arr) - np.min(arr))
    elif norm=='rank':
        return rankdata(arr)
    else:
        print('Normalization technique {} not implemented'.format(norm))
        return -1

def extract_individual_counts(confusion_mtx):
    # tn, fp, fn, tp
    return confusion_mtx.ravel()

def compute_th_metric(y_true, y_hat, y_hat_prob, metric = 'auc'):
    if metric == 'auc':
        if np.unique(y_true).size > 2:
            return roc_auc_score(y_true,y_hat_prob, multi_class = 'ovr')
        else:
            return roc_auc_score(y_true, y_hat_prob)
    elif  metric == 'aucpr':
        return average_precision_score(y_true, y_hat_prob)
    elif metric == 'f1':
        return f1_score(y_true,y_hat)
    elif metric == 'acc':
        return accuracy_score(y_true, y_hat)
    elif metric == "balanced_acc":
        return balanced_accuracy_score(y_true, y_hat)
    elif  metric == 'precision_score':
        return precision_score(y_true, y_hat)
    else:
        return -1
        
    
def compute_metrics(y_true, y_hat, y_hat_prob, label='', suffix='', get_all=False):
    
    cf_mtx = confusion_matrix(y_true, y_hat)
    
    if np.unique(y_true).size > 2:
        auc = roc_auc_score(y_true,y_hat_prob, multi_class = 'ovr')
    else:
        auc = roc_auc_score(y_true, y_hat_prob)
    
    aucpr = average_precision_score(y_true, y_hat_prob)    
    
    tn, fp, fn, tp = extract_individual_counts(cf_mtx)

    surv = tn + fp
    recid = tp + fn

    acc = (tp + tn) / (tp + tn + fp + fn)
    fpr = fp / surv
    fnr = fn / recid
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2*prec*recall)/(prec+recall)
    forate = fn / (fn + tn)
    fdrate = fp / (fp + tp)

    if get_all:
        return pd.DataFrame(data=[[acc, auc, aucpr, fpr, fnr, forate, fdrate, prec, recall, f1]],
                            columns=[col_name + suffix for col_name in
                                     ['Acc', 'AUC', 'AUCPR','FPR', 'FNR', 'for', 'fdr', 'precision', 'recall', 'f1']],
                            index=[label])
    else:
        return pd.DataFrame(data=[[acc, auc, aucpr, f1, fpr, fnr]],
                            columns=[col_name + suffix
                                     for col_name in ['Acc', 'AUC', 'AUCPR', 'f1', 'FPR', 'FNR']], index=[label])
            
def compute_all_scores(dataset, bbox_clf, auditor_clf, INCLUDE_BBOX_OUTPUT = False):
    
    # Bbox Predictions
    y_pred_test = bbox_clf.predict(dataset.X_test)
    y_pred_prob_test = bbox_clf.predict_proba(dataset.X_test)[:,1]    
    bbox_confidence = computeConfidence(bbox_clf, dataset.X_test)
    
    # Bbox Errors
    bbox_errors_test = (y_pred_test!=dataset.y_test).astype(int)    
     
    X_test = dataset.X_test
    X_train = dataset.X_train
    
    # If set, appends bbox model's prediction output to X_train and X_test   
    if INCLUDE_BBOX_OUTPUT:        
        X_train_model_output = bbox_clf.predict_proba(X_train)
        X_test_model_output = bbox_clf.predict_proba(X_test)
        
        X_train = np.concatenate([X_train, X_train_model_output],axis=1)
        X_test = np.concatenate([X_test, X_test_model_output],axis=1)    
    
    # Auditor
    bbox_errors_pred_test = auditor_clf.predict(X_test)
    pred_prob_error, aleatoric_uncertainty, epistemic_uncertainty = get_aleatoric_epistemic_uncertainities(auditor_clf, X_test)
    total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
    
    # Neurips 2018 baseline
    trust_scores = computeTrustScore(X_train = X_train, 
                                     y_train = dataset.y_train, 
                                     X_test = X_test,
                                     y_test_pred = y_pred_test)
    
    test_df_all_scores = pd.DataFrame(dict(aleatoric_uncertainty=aleatoric_uncertainty,
                                           epistemic_uncertainty = epistemic_uncertainty,
                                           total_uncertainty = total_uncertainty,
                                           pred_prob_error = pred_prob_error,
                                           bbox_confidence = bbox_confidence,
                                           trust_scores = trust_scores,
                                           is_bbox_error = bbox_errors_test,
                                           bbox_errors_pred = bbox_errors_pred_test,
                                           y_pred = y_pred_test,
                                           y_pred_prob = y_pred_prob_test,
                                           y = dataset.y_test,
                                           test_group_membership = dataset.test_group_membership,
                                           isOOD = dataset.isOOD,
                                           #isNoisy = dataset.isNoisy
                                            ))
    
    test_df_all_scores['prob_error + epis + alea'] = normalize(pred_prob_error, norm='min-max') + normalize(epistemic_uncertainty, norm='min-max')+ normalize(aleatoric_uncertainty, norm='min-max')    
    test_df_all_scores['Riskscore'] = normalize(pred_prob_error + epistemic_uncertainty + aleatoric_uncertainty, norm='min-max')    
    
    return test_df_all_scores
    
