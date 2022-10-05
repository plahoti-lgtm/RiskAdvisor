import os, sys
import numpy as np
from utils.training_helpers import loadTrainedModelsDataset, computeBBoxMetrics, computeAuditorMetrics
from utils.metrics import compute_all_scores
from multiprocessing import Pool

from utils.visualization import plotDetectTrustworthy, plotDetectSuspicious, visualizeAbstainPlot, plotDensityPlots, plotReliabilityCurve, plot_AUCROC_Curve_Auditor, plotPrecisionRejectionCurve

# EXPERIMENT-SPECIFIC CONFIG
dataset_names = ['uci_adult_OOD','law_school'] #['heart','wine','uci_adult','law_school_OOD'] #'uci_adult_OOD','law_school',] #,'wine','uci_adult_OOD','law_school_OOD','uci_adult', 'law_school']

BBox_names = ['LogisticRegression','Neural Net','RandomForest','SVM'] 
#Auditor_name = 'EnsembleOfCBTs' #['EnsembleOfCBTs', 'EnsembleOfGBTs']

scoring = None 
gridSearchCV = True

if sys.argv[1].lower() == 'cbt':
    Auditor_name = 'EnsembleOfCBTs'
elif sys.argv[1].lower() == 'gbt':
    Auditor_name = 'EnsembleOfGBTs'
else:
    print('Undefined auditor ................... ')
    
INCLUDE_BBOX_OUTPUT = sys.argv[2].lower() == 'true'

experiment_FLAG  = '{}'.format(scoring)  

score_types = ['bbox_confidence', 'trust_scores','pred_prob_error','aleatoric_uncertainty','epistemic_uncertainty','total_uncertainty','Riskscore']

def do(args):    
    dataset_name, BBox_name, Auditor_name, experiment_FLAG, gridSearchCV, score_types = args    
    
    run_process_results(dataset_name = dataset_name,
                   BBox_name = BBox_name,
                   Auditor_name = Auditor_name,
                   experiment_FLAG = experiment_FLAG,
                   gridSearchCV = gridSearchCV,
                   score_types = score_types
                   )

def run_process_results(dataset_name, BBox_name, Auditor_name, experiment_FLAG, gridSearchCV, score_types):
    
    results_base_dir = './results/{}/{}/{}/{}'.format(dataset_name, BBox_name, Auditor_name, experiment_FLAG)
    
    if gridSearchCV:
        dataset, bbox_clf_CV, auditor_clf_CV = loadTrainedModelsDataset(results_base_dir)
        if BBox_name == 'AutoML':
            bbox_clf = bbox_clf_CV
        else:
            bbox_clf = bbox_clf_CV.best_estimator_
            bbox_params = bbox_clf_CV.best_params_
            print('Best BBox: {}-{} : {}'.format(BBox_name, dataset_name, bbox_params))
        
        if Auditor_name in ['EnsembleOfGBTs','EnsembleOfCBTs']:
            auditor_clf = auditor_clf_CV
            auditor_params = []
            for name, est in auditor_clf.named_estimators_.items():
                auditor_params.append((name, est.best_params_))
        else:
            auditor_clf = auditor_clf_CV.best_estimator_
            auditor_params = auditor_clf_CV.best_params_
        
        print('Best Auditor: {}-{} : {}'.format(Auditor_name, dataset_name, auditor_params))
        
    else:
        dataset, bbox_clf, auditor_clf = loadTrainedModelsDataset(results_base_dir)
    
    test_df_all_scores = compute_all_scores(dataset, bbox_clf, auditor_clf, INCLUDE_BBOX_OUTPUT)
        
    output_file_path = os.path.join(results_base_dir,'test_df_all_scores.csv')
    print('Writing results to {}'.format(output_file_path))
    with open(output_file_path, mode="w") as output_file:
        test_df_all_scores.to_csv(output_file,header=True)
        output_file.close()

    
    _score_types = ['bbox_confidence', 'trust_scores','pred_prob_error', 'aleatoric_uncertainty', 'epistemic_uncertainty','total_uncertainty','Riskscore']        
    savefig_path = os.path.join(results_base_dir,'Prediction_Rejection_Curve.png')
    plotPrecisionRejectionCurve(test_df_all_scores, _score_types, orderby_ascending=[True, True, False, False, False, False, False], savefig_path = savefig_path)
    
    savefig_path = os.path.join(results_base_dir,'Out-of-Distribution_AUCROC_Curve.png')
    plot_AUCROC_Curve_Auditor(test_df_all_scores, plottype='isOOD', savefig_path= savefig_path)
     
    metrics = ['acc','f1','aucpr']
    for metric in metrics:
        savefig_path = os.path.join(results_base_dir,'Abstained_vs_{}.png'.format(metric))
        visualizeAbstainPlot(test_df_all_scores, score_types, orderby_ascending = [True, True, False, False, False, False, False], metric = metric, savefig_path = savefig_path)
        print('Plot saved to {}'.format(savefig_path))
        group_values = test_df_all_scores.test_group_membership.unique()
        for group in group_values:
            group_ixs = np.where(test_df_all_scores.test_group_membership.values == group)[0]
            savefig_path = os.path.join(results_base_dir,'Abstained_vs_{}_{}.png'.format(metric, group))
            visualizeAbstainPlot(test_df_all_scores, score_types, orderby_ascending = [True, True, False, False, False, False, False], metric = metric,
                                 filter_ixs = group_ixs, savefig_path = savefig_path, ylabel='{} ({})'.format(metric, group))
            print('Plot saved to {}'.format(savefig_path))
    
    _score_types = ['bbox_confidence', 'trust_scores','epistemic_uncertainty','Riskscore']
    for score_type in _score_types:        
        savefig_path = os.path.join(results_base_dir,'kdeplot_{}_vs_group.png'.format(score_type))
        plotDensityPlots(test_df_all_scores, score_type, hue_col = 'test_group_membership', savefig_path=savefig_path, hue_name = dataset.protected_variable)
        print('Plot saved to {}'.format(savefig_path))
        
    _score_types = ['bbox_confidence', 'trust_scores','pred_prob_error','Riskscore']
    for score_type in _score_types:
        savefig_path = os.path.join(results_base_dir,'kdeplot_{}_vs_error.png'.format(score_type))
        plotDensityPlots(test_df_all_scores, score_type, hue_col = 'is_bbox_error', savefig_path=savefig_path, hue_name='BBox Error')
        print('Plot saved to {}'.format(savefig_path))
    
if __name__ == '__main__':
    if __name__ == '__main__':
        pool = Pool(processes=32)
        args_queue = []
        for BBox_name in BBox_names:
            for dataset_name in dataset_names:
                args_queue.append((dataset_name, BBox_name, Auditor_name, experiment_FLAG, gridSearchCV, score_types))
        pool.map(do, args_queue)
   
