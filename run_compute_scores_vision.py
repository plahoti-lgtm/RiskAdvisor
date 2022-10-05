import tensorflow as tf
import joblib
import os,sys
import numpy as np
import pandas as pd

from utils.uncertainty import get_aleatoric_epistemic_uncertainities, computeTrustScore

from utils.metrics import normalize

dataset_name = sys.argv[1] #'cifar10'
BBox_name = sys.argv[2] #'ResNet50'

if sys.argv[3].lower() == 'cbt':
    Auditor_name = 'EnsembleOfCBTs'
elif sys.argv[3].lower() == 'gbt':
    Auditor_name = 'EnsembleOfGBTs'
else:
    print('Undefined auditor ................... ')
    
scoring = None

gridSearchCV = True
INCLUDE_BBOX_OUTPUT = True

# TODO
experiment_FLAG = '{}'.format(scoring)
        
if INCLUDE_BBOX_OUTPUT:
    experiment_FLAG += 'InclBBoxOutput' #'{}'.format(scoring)            

trained_bbox_model_data_path = './results/{}/{}/EnsembleOfCBTs/NoneInclBBoxOutput/'.format(dataset_name, BBox_name)

results_base_dir = './results/{}/'.format(dataset_name)    
results_dir = os.path.join(results_base_dir,'{}/{}/{}'.format(BBox_name, Auditor_name, experiment_FLAG))    
if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

print('Loading BBox Models at {}'.format(trained_bbox_model_data_path))

bbox_clf = tf.keras.models.load_model(os.path.join(trained_bbox_model_data_path, 'trained_bbox_model'))
       
output_file_path = os.path.join(results_dir,'trained_auditor_model')

if gridSearchCV:    
    auditor_clf_CV = joblib.load(output_file_path)
    if Auditor_name in ['EnsembleOfGBTs','EnsembleOfCBTs']:
        auditor_clf = auditor_clf_CV
        auditor_params = []
        for name, est in auditor_clf.named_estimators_.items():
            auditor_params.append((name, est.best_estimator_))
        Auditor = (Auditor_name, auditor_params)            
    else:
        auditor_clf = auditor_clf_CV.best_estimator_  
        auditor_params = auditor_clf_CV.best_params_
        Auditor = (Auditor_name, auditor_params)
else:
    auditor_clf_CV = joblib.load(output_file_path)
    
print(auditor_clf.named_estimators_.items())

# Load Data and BBOx Models

X_train = np.load(os.path.join(trained_bbox_model_data_path,'X_train.npy'))
X_test = np.load(os.path.join(trained_bbox_model_data_path,'X_test.npy'))        

X_train_xgb = np.load(os.path.join(trained_bbox_model_data_path,'X_train_xgb.npy'))
X_test_xgb = np.load(os.path.join(trained_bbox_model_data_path,'X_test_xgb.npy')) 

if INCLUDE_BBOX_OUTPUT:
                
    print('Augmenting Auditor input with bbox predictions')                       

    X_train_model_output = bbox_clf.predict(X_train)
    X_test_model_output = bbox_clf.predict(X_test)

    X_train_xgb = np.concatenate([X_train_xgb, X_train_model_output],axis=1)
    X_test_xgb = np.concatenate([X_test_xgb, X_test_model_output],axis=1)

y_train_onehot = np.load(os.path.join(trained_bbox_model_data_path,'y_train_onehot.npy'))
y_test_onehot = np.load(os.path.join(trained_bbox_model_data_path,'y_test_onehot.npy'))

y_train = np.load(os.path.join(trained_bbox_model_data_path,'y_train.npy'))
y_test = np.load(os.path.join(trained_bbox_model_data_path,'y_test.npy'))

y_train_pred = np.load(os.path.join(trained_bbox_model_data_path,'y_train_pred.npy'))
y_test_pred = np.load(os.path.join(trained_bbox_model_data_path,'y_test_pred.npy'))

y_train_errors = np.load(os.path.join(trained_bbox_model_data_path,'y_train_errors.npy'))
y_test_errors = np.load(os.path.join(trained_bbox_model_data_path,'y_test_errors.npy'))

isOOD = np.load(os.path.join(trained_bbox_model_data_path,'isOOD.npy'))

# Load Auditor Models
print('Loading Auditor Models at {}'.format(results_dir))

y_train_errors_pred = np.load(os.path.join(results_dir,'y_train_errors_pred.npy'))
y_test_errors_pred = np.load(os.path.join(results_dir,'y_test_errors_pred.npy'))

y_train_errors_pred_prob = np.load(os.path.join(results_dir,'y_train_errors_pred_prob.npy'))
y_test_errors_pred_prob = np.load(os.path.join(results_dir,'y_test_errors_pred_prob.npy'))


# MCP
y_train_pred_prob = bbox_clf.predict(X_train)
y_test_pred_prob = bbox_clf.predict(X_test)
bbox_confidence = bbox_clf.predict(X_test).max(axis=1)    

np.save(os.path.join(trained_bbox_model_data_path,'y_train_pred_prob'),y_train_pred_prob)
np.save(os.path.join(trained_bbox_model_data_path,'y_test_pred_prob'),y_test_pred_prob)

# Neurips 2018 baseline
# Trust score is undefined for datasets with out-of-distribution classes as it expects at least one point with given class label
if dataset_name in ['mnist_ood', 'f_mnist_ood']:    
    trust_scores = np.array([-1]*len(y_test))
else:
    trust_scores = computeTrustScore(X_train = X_train_xgb, 
                                 y_train = y_train, 
                                 X_test = X_test_xgb,
                                 y_test_pred = y_test_pred)                   

print('Trust score:{}'.format(trust_scores))                                      

pred_prob_error, aleatoric_uncertainty, epistemic_uncertainty = get_aleatoric_epistemic_uncertainities(auditor_clf, X_test_xgb)
total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
Riskscore = normalize(pred_prob_error + epistemic_uncertainty + aleatoric_uncertainty, norm='min-max')    


test_df_all_scores = pd.DataFrame(dict(aleatoric_uncertainty=aleatoric_uncertainty,
                                       epistemic_uncertainty = epistemic_uncertainty,
                                       total_uncertainty = total_uncertainty,
                                       pred_prob_error = pred_prob_error,
                                       bbox_confidence = bbox_confidence,
                                       Riskscore = Riskscore,
                                       trust_scores = trust_scores,
                                       is_bbox_error = y_test_errors,
                                       bbox_errors_pred = y_test_errors_pred,
                                       y_pred = y_test_pred,
                                       y = y_test,
                                       isOOD = isOOD
                                        ))

test_df_all_scores['prob_error + epis + alea'] = normalize(pred_prob_error, norm='min-max') + normalize(epistemic_uncertainty, norm='min-max')+ normalize(aleatoric_uncertainty, norm='min-max')    
test_df_all_scores['Riskscore'] = normalize(pred_prob_error + epistemic_uncertainty + aleatoric_uncertainty, norm='min-max')    

output_file_path = os.path.join(results_dir,'test_df_all_scores.csv')
print('Writing results to {}'.format(output_file_path))
with open(output_file_path, mode="w") as output_file:
    test_df_all_scores.to_csv(output_file,header=True)
    output_file.close()
        