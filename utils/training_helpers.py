import os, sys
import numpy as np
import pandas as pd
from utils.metrics import compute_metrics

import joblib

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
 
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from catboost import CatBoostClassifier

from sklearn.calibration import CalibratedClassifierCV

from data_utils import preprocessDataset

def mainTrainer(X_train,
                y_train,
                X_test,
                y_test,
                test_group_membership,
                BBox,
                Auditor,
                gridSearchCV = False,
                label = '',
                scoring = None,
                bbox_model_file_path = None,
                INCLUDE_BBOX_OUTPUT = False):

    Bbox_name, bbox_params = BBox
    Auditor_name, auditor_params = Auditor
    
     # Fitting BBox Model
    print('Training Data Size ...{}'.format(X_train.shape))
    if bbox_model_file_path:
        bbox_clf = joblib.load(bbox_model_file_path)
    else:        
        if gridSearchCV:
            bbox_clf = train_BBoxModel_CV(X_train, y_train, name=Bbox_name, param_grid=bbox_params, scoring = scoring)
        else:
            bbox_clf = train_BBoxModel(X_train, y_train, name=Bbox_name, params=bbox_params)
    
    print('BBox Model fitted...')
    
    y_pred_prob_test = bbox_clf.predict_proba(X_test)[:,1] 
    y_pred_test = bbox_clf.predict(X_test)
    y_pred_train = bbox_clf.predict(X_train)
    
    bbox_results_df = computeBBoxMetrics(y_test,
                                         y_pred_test,
                                         y_pred_prob_test,
                                         test_group_membership,
                                         label = label,
                                         get_all=False)
    
    print(bbox_results_df)
    
    # Fitting Auditor
    error_train = (y_pred_train != y_train).astype(int)    
    print('{} Training Error: {}'.format(Bbox_name, error_train.mean()))
    
    error_test = (y_pred_test != y_test).astype(int)
    print('{} Testing Error: {}'.format(Bbox_name, error_test.mean()))
    
    # If set, appends bbox model's prediction output to X_train and X_test
    if INCLUDE_BBOX_OUTPUT:        
        X_train_model_output = bbox_clf.predict_proba(X_train)
        X_test_model_output = bbox_clf.predict_proba(X_test)
        
        X_train = np.concatenate([X_train, X_train_model_output],axis=1)
        X_test = np.concatenate([X_test, X_test_model_output],axis=1)
    
    if gridSearchCV:
        auditor_clf = train_Auditor_CV(X_train, 
                                error_train,
                                name = Auditor_name,
                                param_grid=auditor_params,
                                scoring = scoring)
    else:
        auditor_clf = train_Auditor(X_train, 
                                error_train,
                                name = Auditor_name,
                                params=auditor_params)
    
    print('Auditor Model fitted...')
    
    y_auditory_pred_prob_test = auditor_clf.predict_proba(X_test)[:,1]
    error_pred_test = auditor_clf.predict(X_test)
    
    
    auditor_results_df = computeAuditorMetrics(error_test,
                                               error_pred_test,
                                               y_auditory_pred_prob_test,
                                               label = label)

    print(auditor_results_df)
    
    return bbox_clf, auditor_clf, bbox_results_df, auditor_results_df

def train_BBoxModel(X_train, y_train, params, name='LogisticRegression'):
    '''Training a BBox Classifier'''
    if name=='Neural Net':
        _bbox_clf = MLPClassifier(**params)
    elif name=='LogisticRegression':
        _bbox_clf = LogisticRegression(**params)
    elif name=='NaiveBayes':
        _bbox_clf = GaussianNB(**params)
    elif name=='RandomForest':
        _bbox_clf = RandomForestClassifier(**params)
    elif name == 'GradientBoosting':
        _bbox_clf = GradientBoostingClassifier(**params)
    elif name=='SVM':
        _bbox_clf = SVC(**params, probability = True)
    else:
        print('BBox Model {} Not Implemented'.format(name))
        return -1

    _bbox_clf.fit(X_train, y_train)
    
    return _bbox_clf

def train_BBoxModel_CV(X_train, y_train, param_grid, name='LogisticRegression', cv = 5, n_jobs = -1, scoring = None):
    '''Training a BBox Classifier'''
    if name=='Neural Net':
        est = MLPClassifier()            
    elif name=='LogisticRegression':
        est = LogisticRegression()
    elif name=='NaiveBayes':
        est = GaussianNB()
    elif name=='RandomForest':
        est = RandomForestClassifier()
    elif name == 'GradientBoosting':
        est = GradientBoostingClassifier()
    elif name=='SVM':
        est = SVC(probability = True)
    else:
        print('BBox Model {} Not Implemented'.format(name))
        return -1
    
    _bbox_clf = GridSearchCV(estimator = est,
                        param_grid = param_grid,
                        cv = cv,
                        n_jobs=n_jobs,
                        scoring = scoring)    
    _bbox_clf.fit(X_train, y_train)
    
    return _bbox_clf

def train_Auditor(X_train, error_train, params, name='RandomForest'):
    '''Training an Auditor to predict BBox's classification errors'''
    if name == 'RandomForest':
        _auditor_clf = RandomForestClassifier(**params)
    elif name == 'ExtraTrees':
        _auditor_clf = ExtraTreesClassifier(**params)
    elif name == 'GradientBoostingTrees':
        _auditor_clf = GradientBoostingClassifier(**params)
    elif name in ['EnsembleOfModels', 'EnsembleOfRFs', 'EnsembleOfGBTs', 'EnsembleOfCBTs']:
        #TODO fetch estimators from params
        # e.g., [('lr', clf1),  ('rf', clf2), ('gnb', clf3), ('svm', clf4), ('gbt',clf5)],
        _auditor_clf = VotingClassifier(estimators = params, 
                        voting='soft',
                        weights=None)    
    else:
        print('Auditor Model {} Not Implemented'.format(name))
        return -1
    
    _auditor_clf.fit(X_train, error_train)
    return _auditor_clf

def train_Auditor_CV(X_train, error_train, param_grid, name='RandomForest', cv = 5, n_jobs = -1, scoring = None):
    '''Training a BBox Classifier'''
    if name=='RandomForest':
        est = RandomForestClassifier()
    elif name == 'ExtraTrees':
        est = ExtraTreesClassifier()
    elif name == 'GradientBoostingTrees':
        est = GradientBoostingClassifier()
    elif name in ['EnsembleOfModels', 'EnsembleOfRFs', 'EnsembleOfGBTs', 'EnsembleOfCBTs']:        
        ensemble_clf = buildEnsemble(param_grid = param_grid, cv = cv, n_jobs = n_jobs, scoring = scoring)
        ensemble_clf.fit(X_train, error_train)
        return ensemble_clf
    else:
        print('Auditor Model {} Not Implemented'.format(name))
        return -1
    
    _auditor_clf = GridSearchCV(estimator = est,
                            param_grid = param_grid,
                            cv = cv,
                            n_jobs=n_jobs,
                            scoring = scoring)
    _auditor_clf.fit(X_train, error_train)
    
    return _auditor_clf

def buildEnsemble(param_grid=None, cv = 3, n_jobs=-1, scoring = None):
    est_names = param_grid['est_names']
    est_params = param_grid['est_params']

    estimators= []
    i = 1
    for name, param_grid in zip(est_names, est_params):
        if name=='lr':
            est = LogisticRegression()
        elif name=='gnb':
            est = GaussianNB()
        elif name=='rf':
            est = RandomForestClassifier()
        elif name == 'svm':
            est = SVC(probability=True)
        elif name == 'gbt':
            est = GradientBoostingClassifier()
        elif name == 'cbt':
            est = CatBoostClassifier(loss_function='Logloss', verbose=100, boosting_type = 'Plain', bootstrap_type ='Bernoulli',
                                     posterior_sampling = True, 
                                     #auto_class_weights = 'Balanced', 
                                     #max_ctr_complexity = 3, 
                                     used_ram_limit = '20gb')
        else:
            print('Model {} Not Implemented in Ensemble'.format(name))     
            continue;
        
        _clf = GridSearchCV(estimator = est,
                    param_grid = param_grid,
                    cv = cv,
                    n_jobs=n_jobs,
                    scoring = scoring)
        
        # Platt Scaling
        # clf = CalibratedClassifierCV(_clf, cv=3, method='sigmoid')
        
        # e.g., [('lr', clf1),  ('rf', clf2), ('gnb', clf3), ('svm', clf4), ('gbt',clf5)],
        i+=1
        estimators.append(('{}-{}'.format(name,i),_clf))        
    
    #Constructing ensemble 
    ensemble_clf = VotingClassifier(estimators= estimators, 
                        voting='soft',
                        weights=None)
    
    return ensemble_clf

def computeBBoxMetrics(y_test,
                       y_pred_test,
                       y_pred_prob_test,
                       test_group_membership,
                       label,
                       get_all=False):
    '''Computes BBox Metrics.
    
    args:
        y_true: np.array
        y_pred_test: np.array
        label: string
        suffix: suffix to be added at the end of metric name (e.g., _s or _ns)
        
    returns:
        bbox_results_df
    '''
    
    bbox_results_df = compute_metrics(y_test, y_pred_test, y_pred_prob_test, label=label, suffix='', get_all=get_all)
        
    groups = np.unique(test_group_membership)
    
    for group in groups:
        group_ixs = np.where(test_group_membership==group)[0]
        y_hat = y_pred_test[group_ixs]
        y_true = y_test[group_ixs]
        y_hat_prob = y_pred_prob_test[group_ixs]
        suffix = '_{}'.format(group)
        
        temp_df = compute_metrics(y_true, y_hat, y_hat_prob, label=label, suffix=suffix, get_all=get_all)
        bbox_results_df = pd.concat([bbox_results_df,temp_df],axis=1)
    return bbox_results_df

def computeAuditorMetrics(error_test,
                          error_pred_test,
                          error_pred_test_prob,
                          label
                         ):
    '''Compute Auditor Metrics.'''
    auditor_results_df = compute_metrics(error_test, error_pred_test, error_pred_test_prob, label='AUD-{}'.format(label), get_all=True)
    return auditor_results_df

def dumpTrainedModelsDataset(results_dir, bbox_clf, auditor_clf, dataset):       
        
    print('Dumping Trained Models at {}'.format(results_dir))
    
    try:    
        output_file_path = os.path.join(results_dir,'trained_bbox_model')
        joblib.dump(bbox_clf, output_file_path)
            
        output_file_path = os.path.join(results_dir,'trained_auditor_model')
        joblib.dump(auditor_clf, output_file_path)
        
        output_file_path = os.path.join(results_dir,'dataset')
        joblib.dump(dataset, output_file_path)

        print('Dump Successful ...')
    except:
        print("Unexpected error:{} at {}".format(sys.exc_info()[0], results_dir))
        raise
        
def loadTrainedModelsDataset(results_dir):
    print('Loading Trained Models from {}'.format(results_dir))
    
    try:
        bbox_model_file_path = os.path.join(results_dir,'trained_bbox_model')
        bbox_clf = joblib.load(bbox_model_file_path)
            
        auditor_model_file_path = os.path.join(results_dir,'trained_auditor_model')
        auditor_clf = joblib.load(auditor_model_file_path)
        
        dataset_object_file_path = os.path.join(results_dir,'dataset')
        dataset = joblib.load(dataset_object_file_path)
        
        return dataset, bbox_clf, auditor_clf
    except:
        print("Unexpected error:{} at {}".format(sys.exc_info()[0], results_dir))
        raise
        
    

def run_experiment(dataset_name, BBox, Auditor, experiment_FLAG, gridSearchCV, scoring, flagNoisy = False, bbox_model_file_path = None, INCLUDE_BBOX_OUTPUT = False):
    
    (Bbox_name, bbox_params) = BBox
    (Auditor_name, auditor_params) = Auditor
    label = '{}{}'.format(dataset_name, Bbox_name)
    
    dataset_base_dir = './data/{}/'.format(dataset_name)
    results_base_dir = './results/{}/'.format(dataset_name)
    
    results_dir = os.path.join(results_base_dir,'{}/{}/{}'.format(Bbox_name, Auditor_name, experiment_FLAG))    
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    
    dataset = preprocessDataset.PreprocessDataset(dataset_name, dataset_base_dir)
    dataset.loadData(flagNoisy)
    
    
    bbox_clf, auditor_clf, bbox_results_df, auditor_results_df = mainTrainer(dataset.X_train, 
                                                                           dataset.y_train, 
                                                                           dataset.X_test, 
                                                                           dataset.y_test,
                                                                           dataset.test_group_membership,
                                                                           BBox = BBox,
                                                                           Auditor = Auditor,
                                                                           label = label,
                                                                           gridSearchCV = gridSearchCV,
                                                                           scoring = scoring,
                                                                           bbox_model_file_path = bbox_model_file_path,
                                                                           INCLUDE_BBOX_OUTPUT = INCLUDE_BBOX_OUTPUT)
    
    
    print('Model Training Completed ...')    
    
    print(bbox_results_df)
    print(auditor_results_df)
    
    dumpTrainedModelsDataset(results_dir, bbox_clf, auditor_clf, dataset)
    
    output_file_path = os.path.join(results_dir,'bbox_results.csv')
    with open(output_file_path, mode="w") as output_file:
        bbox_results_df.to_csv(output_file,header=True)
        output_file.close()
        
    output_file_path = os.path.join(results_dir,'auditor_results.csv')
    with open(output_file_path, mode="w") as output_file:
        auditor_results_df.to_csv(output_file,header=True)
        output_file.close()
    
    print('Successfull...')