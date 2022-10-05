# -*- coding: utf-8 -*-
import numpy as np
from utils.training_helpers import run_experiment
from multiprocessing import Pool
import sys

# EXPERIMENT-SPECIFIC CONFIG
dataset_name = sys.argv[1] #['heart', 'uci_adult', 'law_school']
flagNoisy = False
scoring_s = [None]

gridSearchCV = True
INCLUDE_BBOX_OUTPUT = True
    
TRAIN_BBOX = False

BBox_s = [
        ('LogisticRegression', {"penalty":["l2"],
                                  'solver': ['lbfgs']}),
          ('Neural Net', {'hidden_layer_sizes': [(32,16)],
                         'activation': ['relu'],
                         'solver': ['sgd'],
                         'batch_size' : [64],
                         'learning_rate_init' : [0.001, 0.01, 0.1]
                         }),
          ('RandomForest', {'n_estimators': [1000],
                            'max_features': ['sqrt'],
                            'bootstrap': [True],
                            'max_samples':[0.6],
                            'class_weight':['balanced'],
                            'max_depth' : [5]
                            }),
          ('SVM',{'class_weight':['balanced'], 'probability':[True]}),
          ]


Auditor_s = [
        # Ensemble of Stochastic Gradient Boosted Trees - SKlearn implementation
        ('ensembleofgbts', 
         {'est_names': ['gbt','gbt','gbt','gbt','gbt','gbt','gbt','gbt','gbt','gbt'],
          'est_params':[{"n_estimators":[100], 'subsample':[0.25,0.5,0.75], 'max_depth':[3,4,5,6], 'random_state':[1], 'learning_rate':[0.1,0.01,0.001]},
                        {"n_estimators":[100], 'subsample':[0.25,0.5,0.75], 'max_depth':[3,4,5,6], 'random_state':[2], 'learning_rate':[0.1,0.01,0.001]},
                        {"n_estimators":[100], 'subsample':[0.25,0.5,0.75], 'max_depth':[3,4,5,6], 'random_state':[3], 'learning_rate':[0.1,0.01,0.001]},
                        {"n_estimators":[100], 'subsample':[0.25,0.5,0.75], 'max_depth':[3,4,5,6], 'random_state':[4], 'learning_rate':[0.1,0.01,0.001]},
                        {"n_estimators":[100], 'subsample':[0.25,0.5,0.75], 'max_depth':[3,4,5,6], 'random_state':[5], 'learning_rate':[0.1,0.01,0.001]},
                        {"n_estimators":[100], 'subsample':[0.25,0.5,0.75], 'max_depth':[3,4,5,6], 'random_state':[6], 'learning_rate':[0.1,0.01,0.001]},
                        {"n_estimators":[100], 'subsample':[0.25,0.5,0.75], 'max_depth':[3,4,5,6], 'random_state':[7], 'learning_rate':[0.1,0.01,0.001]},
                        {"n_estimators":[100], 'subsample':[0.25,0.5,0.75], 'max_depth':[3,4,5,6], 'random_state':[8], 'learning_rate':[0.1,0.01,0.001]},
                        {"n_estimators":[100], 'subsample':[0.25,0.5,0.75], 'max_depth':[3,4,5,6], 'random_state':[9], 'learning_rate':[0.1,0.01,0.001]},
                        {"n_estimators":[100], 'subsample':[0.25,0.5,0.75], 'max_depth':[3,4,5,6], 'random_state':[10],'learning_rate':[0.1,0.01,0.001]}
                        ]}
          ),
        # Ensemble of Stochastic Gradient Boosted Trees - Catboost implementation
        ('EnsembleOfCBTs', 
                 {'est_names': ['cbt','cbt','cbt','cbt','cbt','cbt','cbt','cbt','cbt','cbt'],
                  'est_params':[{"n_estimators":[100], 'subsample':[0.25,0.5,0.75], 'depth':[3, 4, 5, 6], 'random_seed':[1],'learning_rate':[0.1,0.01,0.001]},
                                {"n_estimators":[100], 'subsample':[0.25,0.5,0.75], 'depth':[3, 4, 5, 6], 'random_seed':[2],'learning_rate':[0.1,0.01,0.001]},
                                {"n_estimators":[100], 'subsample':[0.25,0.5,0.75], 'depth':[3, 4, 5, 6], 'random_seed':[3],'learning_rate':[0.1,0.01,0.001]},
                                {"n_estimators":[100], 'subsample':[0.25,0.5,0.75], 'depth':[3, 4, 5, 6], 'random_seed':[4],'learning_rate':[0.1,0.01,0.001]},
                                {"n_estimators":[100], 'subsample':[0.25,0.5,0.75], 'depth':[3, 4, 5, 6], 'random_seed':[5],'learning_rate':[0.1,0.01,0.001]},
                                {"n_estimators":[100], 'subsample':[0.25,0.5,0.75], 'depth':[3, 4, 5, 6], 'random_seed':[6],'learning_rate':[0.1,0.01,0.001]},
                                {"n_estimators":[100], 'subsample':[0.25,0.5,0.75], 'depth':[3, 4, 5, 6], 'random_seed':[7],'learning_rate':[0.1,0.01,0.001]},
                                {"n_estimators":[100], 'subsample':[0.25,0.5,0.75], 'depth':[3, 4, 5, 6], 'random_seed':[8],'learning_rate':[0.1,0.01,0.001]},
                                {"n_estimators":[100], 'subsample':[0.25,0.5,0.75], 'depth':[3, 4, 5, 6], 'random_seed':[8],'learning_rate':[0.1,0.01,0.001]},
                                {"n_estimators":[100], 'subsample':[0.25,0.5,0.75], 'depth':[3, 4, 5, 6], 'random_seed':[10],'learning_rate':[0.1,0.01,0.001]}
                                ]})
          ]

def do(args):    
    dataset_name, BBox, Auditor, experiment_FLAG, gridSearchCV, scoring, bbox_model_file_path, INCLUDE_BBOX_OUTPUT = args    
    
    run_experiment(dataset_name = dataset_name,
                   BBox = BBox,
                   Auditor = Auditor,
                   experiment_FLAG = experiment_FLAG,
                   gridSearchCV = gridSearchCV,
                   scoring = scoring,
                   bbox_model_file_path = bbox_model_file_path,
                   INCLUDE_BBOX_OUTPUT = INCLUDE_BBOX_OUTPUT
                   )
    
    
if __name__ == '__main__':
    if __name__ == '__main__':        
        if (len(BBox_s) == 1) and (len(BBox_s) == 1) and (len(scoring_s)==1):
            experiment_FLAG  = '{}'.format(scoring_s[0])  
            
            if TRAIN_BBOX:
                trained_bbox_model_data_path = None
            else:
                bbox_model_file_path = './results/{}/{}/{}/{}/trained_bbox_model'.format(dataset_name, BBox_s[0][0], Auditor_s[0][0], experiment_FLAG)                
            if INCLUDE_BBOX_OUTPUT:
                experiment_FLAG += 'InclBBoxOutput'
                
            do((dataset_name, BBox_s[0], Auditor_s[0], experiment_FLAG, gridSearchCV, scoring_s[0], bbox_model_file_path, INCLUDE_BBOX_OUTPUT))
        else:
            pool = Pool(processes=16)
            args_queue = []
            for BBox in BBox_s:
                for Auditor in Auditor_s:
                    for scoring in scoring_s:
                        experiment_FLAG = '{}'.format(scoring)   
                        
                        # If set, trains the black box model, if not loads it from the location
                        if TRAIN_BBOX:
                            trained_bbox_model_data_path = None
                        else:
                            bbox_model_file_path = './results/{}/{}/{}/{}/trained_bbox_model'.format(dataset_name, BBox[0], Auditor[0], experiment_FLAG)
                        if INCLUDE_BBOX_OUTPUT:
                            experiment_FLAG += 'InclBBoxOutput'
                            
                        args_queue.append((dataset_name, BBox, Auditor, experiment_FLAG, gridSearchCV, scoring, bbox_model_file_path, INCLUDE_BBOX_OUTPUT))
            pool.map(do, args_queue)

