import os
import pandas as pd
import numpy as np
from utils.sampling import sample_retrain_wrapper
from utils.training_helpers import loadTrainedModelsDataset, computeBBoxMetrics

def run_sampling_experiment(dataset_name, BBox_name, Auditor_name, experiment_FLAG, sampling_FLAG, gridSearchCV,
                            max_region_size, sampling_techniques,
                            num_runs, num_loops, sampling_percentage):    
    
    model_base_dir = './results/{}/{}/{}/{}'.format(dataset_name, BBox_name, Auditor_name, experiment_FLAG)
    
    results_base_dir = os.path.join('./results/{}/{}/{}/{}'.format(dataset_name, BBox_name, Auditor_name, experiment_FLAG, sampling_FLAG))
    
    if not os.path.isdir(results_base_dir):
        os.makedirs(results_base_dir)
        
    print('Loading Trained Models from {}'.format(model_base_dir))
    
    if gridSearchCV:
        dataset, bbox_clf_CV, auditor_clf_CV = loadTrainedModelsDataset(model_base_dir)
        bbox_clf = bbox_clf_CV.best_estimator_
        bbox_params = bbox_clf_CV.best_params_
        BBox = (BBox_name, bbox_params)
        
        if Auditor_name == 'EnsembleOfGBTs':
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
        dataset, bbox_clf, auditor_clf = loadTrainedModelsDataset(model_base_dir)

    print('Best Auditor: {}-{} : {}'.format(Auditor_name, dataset_name, auditor_params))
   
    sample_retrain_results_df = pd.DataFrame()
    
    for run in np.arange(num_runs):
        for sample_by in sampling_techniques:
            temp_df = sample_retrain_wrapper(dataset,
                                      bbox_clf,
                                      auditor_clf,
                                      BBox,
                                      Auditor,
                                      max_region_size = max_region_size,
                                      sampling_percentage = sampling_percentage,
                                      sample_by = sample_by,
                                      num_loops=num_loops)
            temp_df['run'] = run
            sample_retrain_results_df = pd.concat([sample_retrain_results_df,temp_df])
    
    sample_retrain_results_df = add_baseline_results(dataset, bbox_clf, sample_retrain_results_df, sampling_techniques, label='{}{}'.format(BBox_name,'Baseline'))
    
    print('Sample-Retrain Completed ...')
    
    print('Writing results to file to {}'.format(results_base_dir))
    
    output_file_path = os.path.join(results_base_dir,'sample_retrain_results_max_region_size-{}.csv'.format(max_region_size))
    header= not(os.path.exists(output_file_path))
    with open(output_file_path, mode="a") as output_file:
        sample_retrain_results_df.to_csv(output_file, mode='a', index=False, header=header)
        output_file.close()

    mean_sample_retrain_results_df = sample_retrain_results_df.groupby(by=['sampling','sample_size']).mean().reset_index()
    header= not(os.path.exists(output_file_path))
    output_file_path = os.path.join(results_base_dir,'mean_sample_retrain_results_max_region_size-{}.csv'.format(max_region_size))
    with open(output_file_path, mode="a") as output_file:
        mean_sample_retrain_results_df.to_csv(output_file, mode='a', index=False,header=header)
        output_file.close()

def add_baseline_results(dataset, bbox_clf, mean_sample_retrain_results_df, sampling_techniques, label=''):
    train_size = dataset.y_test.shape[0]
    y_test = dataset.y_test
    y_pred_test = bbox_clf.predict(dataset.X_test)
    y_pred_prob_test = bbox_clf.predict_proba(dataset.X_test)[:,1]
    
    bbox_results_df = computeBBoxMetrics(y_test,
                                         y_pred_test,
                                         y_pred_prob_test,
                                         dataset.test_group_membership,
                                         label = label,
                                         get_all=False)
    
    for sampling in sampling_techniques:
        temp_df = bbox_results_df.copy()
        temp_df['run'] = 0
        temp_df['sampling'] = sampling
        temp_df['sample_size'] = 0
        temp_df['train_size'] = train_size
        mean_sample_retrain_results_df = pd.concat([mean_sample_retrain_results_df,temp_df])  
    
    mean_sample_retrain_results_df = mean_sample_retrain_results_df.sort_values(by='train_size')
    return mean_sample_retrain_results_df