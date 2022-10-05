import numpy as np
import pandas as pd
from collections import Counter, defaultdict

from sklearn.neighbors import KDTree
from sklearn.tree import DecisionTreeRegressor

from TrustScore import trustscore
from utils.training_helpers import mainTrainer
from utils.uncertainty import get_aleatoric_epistemic_uncertainities
from utils.uncertainty import computeConfidence

def normalize(arr):
    ''' Args:
            np.array
        returns:
            array after sum to 1 normalization'''
    return arr/np.sum(arr)

def get_sorted_indices(X, ascending=False):
    """
    Given a array of values, returns a np.array of top k indices (sorted in descending order)
    """
    if ascending:
        return X.argsort()
    else:
        return X.argsort()[::-1]

def getKernelDensityInRegion(X_train, X_test):
    '''returns Kernel Density in Training Data in this region'''
    train_data_tree = KDTree(X_train)
    return train_data_tree.kernel_density(X_test, h=0.1, kernel='gaussian')


def samplePointsFromRegion(regions,
                           prob,
                           sample_size,
                           region_ixs_dict):
    '''
    Return ixs of datapoints drawn from each region proportionate to the input probabilities. 
    Within a region points are drawn uniformly at random.
    
    Args:
        regions: np.array of region names. Should match 
        prob: region probabilities
        sample_size: number of samples to draw
        region_ixs_dict: dict mapping region to ix of datapoints in the region.
    '''
    # Constructs a counter of regions to sample from proportionate to their probabilities
    draw_regions = Counter(np.random.choice(a=regions, size=sample_size, p=prob))
    
    # Within a region points are drawn uniformly at random.
    sampling_ixs = []
    for r, size in draw_regions.items():
        if r not in region_ixs_dict or len(region_ixs_dict[r])==0:
            print('No points in region {} in the Dataset'.format(r))
            continue;
        else:
            sampling_ixs.extend(np.random.choice(a = region_ixs_dict[r],size = size))
    return sampling_ixs

def sample_by_bbox_confidence(bbox_clf, X_ood, sample_size, ascending=True):
    '''
    Returns ixs of points with Lowest (or Highest if ascending=True) BBox Confidence.
    '''
    bbox_confidence = bbox_clf.predict_proba(X_ood).max(axis=1)
    bbox_argsort_ix = get_sorted_indices(bbox_confidence, ascending = ascending)
    sampling_ixs = bbox_argsort_ix[:sample_size]
    return sampling_ixs

def sample_by_density(X_ood, sample_size, ascending=True):
    kd_tree = KDTree(X_ood)
    density = kd_tree.kernel_density(X_ood, h=0.1, kernel='gaussian')
    density_argsort_ix = get_sorted_indices(density, ascending = ascending)
    sampling_ixs = density_argsort_ix[:sample_size]
    return sampling_ixs

def sample_retrain_by_auditor(dataset,
                             bbox_clf,
                             auditor_clf,
                             BBox,
                             Auditor,
                             label = 'epistemic',
                             sample_by = 'epistemic',
                             max_region_size = None,
                             sampling_percentage = 1,
                             num_loops = 10):
    
    N = dataset.X_ood.shape[0]
    sample_size = int((sampling_percentage*N)/100)
    
    X_retrain = dataset.X_train.copy()
    y_retrain = dataset.y_train.copy()
    
    X_unlabeled = dataset.X_ood.copy()
    y_unlabeled = dataset.y_ood.copy()
    
    bbox_results_df = pd.DataFrame()
    
    for loop in np.arange(num_loops):
        
        # Order all points in unlabelled as per their uncertainty and sample
        if max_region_size is None:       
            
            pred_prob_error, aleatoric_uncertainty, epistemic_uncertainty = get_aleatoric_epistemic_uncertainities(auditor_clf, X_unlabeled) 
    
            if sample_by=='pred_prob_error': 
                target_values = pred_prob_error.copy()
            elif sample_by=='aleatoric': 
                target_values = aleatoric_uncertainty.copy()
            elif sample_by=='epistemic': 
                target_values = epistemic_uncertainty.copy()
            
            auditor_argsort_ix = get_sorted_indices(target_values, ascending = False)
            sampling_ixs = auditor_argsort_ix[:sample_size]
        
        # Fit approximated regions and sample from these regions
        else:            
            dtree, approx_regions_values_dict, auditor_region_ixs_dict = get_auditor_approx_regions_values_for_sampling(bbox_clf,
                                                                       auditor_clf,
                                                                       X_unlabeled,
                                                                       max_region_size = max_region_size,
                                                                       sample_by=sample_by)
    
            auditor_regions = list(approx_regions_values_dict.keys())
            auditor_region_values = list(approx_regions_values_dict.values())
    
            # Normalize and sample proportionate to probability
            prob = normalize(auditor_region_values)
            sampling_ixs = samplePointsFromRegion(regions = auditor_regions,
                                   prob = prob,
                                   sample_size = sample_size,
                                   region_ixs_dict = auditor_region_ixs_dict 
                                  )
            
        # Sample and Construct retrain set
        X_retrain = np.concatenate([X_retrain,X_unlabeled[sampling_ixs]]) 
        y_retrain = np.concatenate([y_retrain,y_unlabeled[sampling_ixs]])
        
        # Comment out the following to allow Auditor to Sample with Replacement (so that it can oversample under-represented points)
        # exclude sampling_ixs from X_unlabeled and y_unlabeled (preserve order)
        #X_unlabeled = np.delete(X_unlabeled,sampling_ixs, axis=0)
        #y_unlabeled = np.delete(y_unlabeled,sampling_ixs, axis=0)
        
        print('Retraining ... ')
        
        # Currently, I am sampling from retrained auditor. 
        # Edit the following line and remove auditor_clf to change this behavior
        bbox_clf, auditor_clf, temp_bbox_results_df, _ = mainTrainer(X_retrain,
                                                                   y_retrain,
                                                                   dataset.X_test, 
                                                                   dataset.y_test,
                                                                   dataset.test_group_membership,
                                                                   BBox = BBox,
                                                                   Auditor = Auditor,
                                                                   label=label)
        temp_bbox_results_df['sampling'] = label
        temp_bbox_results_df['sample_size'] = (loop+1)*sampling_percentage
        temp_bbox_results_df['train_size'] = X_retrain.shape[0]
        bbox_results_df = pd.concat([bbox_results_df,temp_bbox_results_df])
        
    return bbox_results_df

def sample_retrain_by_bbox_confidence(dataset,
                                       bbox_clf,
                                       BBox,
                                       Auditor,
                                       sampling_percentage = 1,
                                       label = 'bbox_confidence',
                                       num_loops=10,
                                       ascending=True):
    '''
    Rank unlabeled points by BBox Confidence, sample points with Lowest confidence and re-train BBox.
    '''
    N = dataset.X_ood.shape[0]
    sample_size = int((sampling_percentage*N)/100)
    
    X_retrain = dataset.X_train.copy()
    y_retrain = dataset.y_train.copy()
    
    X_unlabeled = dataset.X_ood.copy()
    y_unlabeled = dataset.y_ood.copy()
    
    bbox_results_df = pd.DataFrame()
    
    for loop in np.arange(num_loops):
        
        # Rank unlabeled points by BBox Uncertainty
        bbox_confidence = computeConfidence(bbox_clf, X_unlabeled)
        bbox_argsort_ix = get_sorted_indices(bbox_confidence, ascending = ascending)
                
        # Sample and Construct retrain set
        sampling_ixs = bbox_argsort_ix[:sample_size]
        X_retrain = np.concatenate([X_retrain,X_unlabeled[sampling_ixs]]) 
        y_retrain = np.concatenate([y_retrain,y_unlabeled[sampling_ixs]])
        
        # exclude sampling_ixs from X_unlabeled and y_unlabeled (preserve order)
        X_unlabeled = np.delete(X_unlabeled,sampling_ixs, axis=0)
        y_unlabeled = np.delete(y_unlabeled,sampling_ixs, axis=0)
        
        # Re-train 
        bbox_clf, _, temp_bbox_results_df, _ = mainTrainer(X_retrain,
                                                                   y_retrain,
                                                                   dataset.X_test, 
                                                                   dataset.y_test,
                                                                   dataset.test_group_membership,
                                                                   BBox = BBox,
                                                                   Auditor = Auditor,
                                                                   label=label)
        temp_bbox_results_df['sampling'] = label
        temp_bbox_results_df['sample_size'] = (loop+1)*sampling_percentage
        temp_bbox_results_df['train_size'] = X_retrain.shape[0]
        bbox_results_df = pd.concat([bbox_results_df,temp_bbox_results_df])
        
    return bbox_results_df

def sample_retrain_by_trust_score(dataset,
                                       bbox_clf,
                                       BBox,
                                       Auditor,
                                       sampling_percentage = 1,
                                       label = 'trust_score',
                                       num_loops=10,
                                       alpha = 0.01,
                                       k = 10,
                                       filtering = 'density',
                                       ascending=True):
    '''
    Rank unlabeled points by Trust Score, sample points with lowest Trust score and re-train BBox.
    '''
    N = dataset.X_ood.shape[0]
    sample_size = int((sampling_percentage*N)/100)
    
    X_retrain = dataset.X_train.copy()
    y_retrain = dataset.y_train.copy()
    
    X_unlabeled = dataset.X_ood.copy()
    y_unlabeled = dataset.y_ood.copy()
    
    bbox_results_df = pd.DataFrame()
    
    # Fitting Trust Model to Training Data Once 
    ## Trust model fitting depends only on training data. BBox_clf's predictions are only needed for evaluation
    trust_model = trustscore.TrustScore(k=k, alpha=alpha, filtering=filtering)
    trust_model.fit(dataset.X_train, dataset.y_train.astype(int))
    
    for loop in np.arange(num_loops):
        
        # Rank unlabeled points by Trust Score
        ## Compute trusts score, given (unlabeled) testing examples and bbox models predictions.
        y_pred_unlabeled = bbox_clf.predict(X_unlabeled)
        trust_scores = trust_model.get_score(X_unlabeled, y_pred_unlabeled.astype(int))    
        trust_scores_argsort_ix = get_sorted_indices(trust_scores, ascending = ascending)
                
        # Sample and Construct retrain set
        sampling_ixs = trust_scores_argsort_ix[:sample_size]
        X_retrain = np.concatenate([X_retrain,X_unlabeled[sampling_ixs]]) 
        y_retrain = np.concatenate([y_retrain,y_unlabeled[sampling_ixs]])
        
        # exclude sampling_ixs from X_unlabeled and y_unlabeled (preserve order)
        X_unlabeled = np.delete(X_unlabeled,sampling_ixs, axis=0)
        y_unlabeled = np.delete(y_unlabeled,sampling_ixs, axis=0)
        
        # Re-train 
        bbox_clf, _, temp_bbox_results_df, _ = mainTrainer(X_retrain,
                                                                   y_retrain,
                                                                   dataset.X_test, 
                                                                   dataset.y_test,
                                                                   dataset.test_group_membership,
                                                                   BBox = BBox,
                                                                   Auditor = Auditor,
                                                                   label=label)
        temp_bbox_results_df['sampling'] = label
        temp_bbox_results_df['sample_size'] = (loop+1)*sampling_percentage
        temp_bbox_results_df['train_size'] = X_retrain.shape[0]
        bbox_results_df = pd.concat([bbox_results_df,temp_bbox_results_df])
        
    return bbox_results_df

def sample_retrain_by_density(dataset,
                              task_results, 
                              auditor_task_results,
                              BBox,
                              Auditor,
                              sampling_percentage = 1,
                              label = 'lowest_density',
                              num_loops=10, ascending=True):
    '''
    Rank unlabeled points by density, sample and re-train BBox.
    '''
    N = dataset.X_ood.shape[0]
    sample_size = int((sampling_percentage*N)/100)
    
    X_retrain = dataset.X_train.copy()
    y_retrain = dataset.y_train.copy()
    
    X_unlabeled = dataset.X_ood.copy()
    y_unlabeled = dataset.y_ood.copy()
    
    bbox_results_df = pd.DataFrame()
    
    # TODO: investigate if this is the right way
    kd_tree = KDTree(dataset.X_train)
    
    for loop in np.arange(num_loops):        

        density = kd_tree.kernel_density(X_unlabeled, h=0.1, kernel='gaussian')
        density_argsort_ix = get_sorted_indices(density, ascending = ascending)
        sampling_ixs = density_argsort_ix[:sample_size]
        
        X_retrain = np.concatenate([X_retrain,X_unlabeled[sampling_ixs]]) 
        y_retrain = np.concatenate([y_retrain,y_unlabeled[sampling_ixs]])
        
        # exclude sampling_ixs from X_unlabeled and y_unlabeled (preserve order)
        #X_unlabeled = np.delete(X_unlabeled,sampling_ixs, axis=0)
        #y_unlabeled = np.delete(y_unlabeled,sampling_ixs, axis=0)
        
         # Re-train 
        _, _, temp_bbox_results_df, _ = mainTrainer(X_retrain,
                                                         y_retrain,
                                                         dataset.X_test, 
                                                         dataset.y_test,
                                                         dataset.test_group_membership,
                                                         BBox = BBox,
                                                         Auditor = Auditor,
                                                         label=label)
        temp_bbox_results_df['sampling'] = label
        temp_bbox_results_df['sample_size'] = (loop+1)*sampling_percentage
        temp_bbox_results_df['train_size'] = X_retrain.shape[0]
        bbox_results_df = pd.concat([bbox_results_df,temp_bbox_results_df])
        
    return bbox_results_df

def sample_retrain_wrapper(dataset,
                          bbox_clf,
                          auditor_clf,
                          BBox,
                          Auditor,
                          sample_by,
                          max_region_size = 0.1,
                          sampling_percentage = 1,
                          num_loops=10):
    '''
    Sample and re-train
    '''
    if sample_by in ['epistemic', 'aleatoric', 'pred_prob_error']:
        print('Sampling via {}'.format(sample_by))
        bbox_results_df = sample_retrain_by_auditor(dataset,
                                 bbox_clf,
                                 auditor_clf,
                                 BBox,
                                 Auditor,
                                 sampling_percentage = sampling_percentage,
                                 num_loops=num_loops,
                                 label = sample_by,
                                 sample_by = sample_by,
                                 max_region_size = max_region_size,
                                 )

    elif sample_by in ['bbox_confidence']:
        print('Sampling via {}'.format(sample_by))
        bbox_results_df = sample_retrain_by_bbox_confidence(dataset,
                                   bbox_clf,
                                   BBox,
                                   Auditor,
                                   sampling_percentage = sampling_percentage,
                                   num_loops=num_loops,
                                   label = 'bbox_confidence',                                       
                                   ascending=True)

    elif sample_by in ['density']:
        print('Sampling via {}'.format(sample_by))
        bbox_results_df = sample_retrain_by_density(dataset,
                          BBox,
                          Auditor,
                          sampling_percentage = sampling_percentage,
                          num_loops=num_loops,
                          label = 'lowest_density',
                          ascending=True)
    elif sample_by in ['trust_score']:
        print('Sampling via {}'.format(sample_by))
        bbox_results_df = sample_retrain_by_trust_score(dataset,
                                   bbox_clf,
                                   BBox,
                                   Auditor,
                                   sampling_percentage = sampling_percentage,
                                   num_loops=num_loops,
                                   label = 'trust_score',                                       
                                   ascending=True)
    elif sample_by in ['baseline']:
        # Sample 100% OOD, and re-train -- Best possible
        sampling_percentage = 100
        
        N = dataset.X_ood.shape[0]
        sample_size = int((sampling_percentage*N)/100)
        
        X_retrain = dataset.X_train.copy()
        y_retrain = dataset.y_train.copy()

        X_unlabeled = dataset.X_ood.copy()
        y_unlabeled = dataset.y_ood.copy()

        bbox_results_df = pd.DataFrame()

        # Sample and Construct retrain set            
        N = X_unlabeled.shape[0]
        sampling_ixs = np.random.choice(N, size=sample_size, replace=False)            
        X_retrain = np.concatenate([X_retrain,X_unlabeled[sampling_ixs]]) 
        y_retrain = np.concatenate([y_retrain,y_unlabeled[sampling_ixs]])
        
        
         # Re-train 
        _, _,temp_bbox_results_df, _ = mainTrainer(X_retrain,
                                                         y_retrain,
                                                         dataset.X_test, 
                                                         dataset.y_test,
                                                         dataset.test_group_membership,
                                                         BBox = BBox,
                                                         Auditor = Auditor,
                                                         label=sample_by)
        temp_bbox_results_df['sampling'] = sample_by
        temp_bbox_results_df['sample_size'] = sampling_percentage
        temp_bbox_results_df['train_size'] = X_retrain.shape[0]
        bbox_results_df = pd.concat([bbox_results_df,temp_bbox_results_df])        

    elif sample_by in ['random']:
        # Rank unlabeled points randomly, sample and re-train.
        N = dataset.X_ood.shape[0]
        sample_size = int((sampling_percentage*N)/100)
        
        X_retrain = dataset.X_train.copy()
        y_retrain = dataset.y_train.copy()

        X_unlabeled = dataset.X_ood.copy()
        y_unlabeled = dataset.y_ood.copy()

        bbox_results_df = pd.DataFrame()

        for loop in np.arange(num_loops):

            # Sample and Construct retrain set            
            N = X_unlabeled.shape[0]
            sampling_ixs = np.random.choice(N, size=sample_size, replace=False)            
            X_retrain = np.concatenate([X_retrain,X_unlabeled[sampling_ixs]]) 
            y_retrain = np.concatenate([y_retrain,y_unlabeled[sampling_ixs]])
            
            # exclude sampling_ixs from X_unlabeled and y_unlabeled (preserve order)
            X_unlabeled = np.delete(X_unlabeled,sampling_ixs, axis=0)
            y_unlabeled = np.delete(y_unlabeled,sampling_ixs, axis=0)

             # Re-train 
            _, _,temp_bbox_results_df, _ = mainTrainer(X_retrain,
                                                             y_retrain,
                                                             dataset.X_test, 
                                                             dataset.y_test,
                                                             dataset.test_group_membership,
                                                             BBox = BBox,
                                                             Auditor = Auditor,
                                                             label=sample_by)
            temp_bbox_results_df['sampling'] = sample_by
            temp_bbox_results_df['sample_size'] = (loop+1)*sampling_percentage
            temp_bbox_results_df['train_size'] = X_retrain.shape[0]
            bbox_results_df = pd.concat([bbox_results_df,temp_bbox_results_df])
    else:
        return -1
    
    return bbox_results_df


def get_auditor_approx_regions_values_for_sampling(bbox_clf,
                                                   auditor_clf,
                                                   X,
                                                   max_region_size = 0.1,
                                                   sample_by='epistemic'):
    
    pred_prob_error, aleatoric_uncertainty, epistemic_uncertainty = get_aleatoric_epistemic_uncertainities(auditor_clf, X)

    if sample_by=='pred_prob_error': 
        target_values = pred_prob_error.copy()
    elif sample_by=='aleatoric': 
        target_values = aleatoric_uncertainty.copy()
    elif sample_by=='epistemic': 
        target_values = epistemic_uncertainty.copy()

    print('Approximating the {} Score of Ensemble Auditor via DTree Regressor'.format(sample_by))

    dtree, approx_regions_values_dict, approx_region_X_ixs_dict, = get_approximated_dtree_and_regions(X,
                                                             target_values = target_values,
                                                             max_region_size = max_region_size)

    print('Good-ness of fit of the approximated DTree ... \nCoefficient of determination R^2 is {:.2f}'.format(dtree.score(X,target_values)))
    
    return dtree, approx_regions_values_dict, approx_region_X_ixs_dict

def get_approximated_dtree_and_regions(X, target_values, max_region_size = 0.1, max_features = None, criterion = 'friedman_mse'):
    '''
    Fits a Decision Tree Regressor to approximate Auditor's estimated epstemic, aleatoric or predicted-error-probability.
    args:
        - X
        - target_values : values to be approximated via regression
        - max_regions
    returns:
        - Approximated Dtree
        - R_p_y_x: dict with keys as Region Id and values as p(y|x) of this region {R_id: p_y_x}
        - region_ix_dict: dict with keys as Region Id and values as a list of X_ix that have membership to this region {R_id: [X_ixs]}
        
    '''  
    
    # Fit a Dtree to Approximate the RF Auditor
    dtree = DecisionTreeRegressor(min_samples_leaf = max_region_size, max_features = max_features, criterion = criterion)
    dtree.fit(X, target_values)
    
    # Dict of Region-ids, and list of X ixs with membership to it. {Region_id : [X_ixs, ... ]}
    X_region_ids = dtree.apply(X)
    region_ix_dict = defaultdict(list)
    for ix, val in enumerate(X_region_ids):
        region_ix_dict[val].append(ix)
        
    # Dict of Region-ids and approximated target values (e.g., approx epistemic uncert.) of this region
    approx_target_values = dtree.predict(X)
    R_approx_target_values_dict = dict(zip(X_region_ids, approx_target_values))
    
    return dtree, R_approx_target_values_dict, region_ix_dict
    
#def getRegionIDs(auditor_clf, X):
#    '''returns 
#        - np.ndarray of regions that each datapoint in X belongs to for each estimator in auditor_clf
#        - a dict of {(str) region: (list) datapoint_ixs} 
#            (where region is concatenation of regions from each decision tree estimator) 
#    '''
#    X_region_ids = auditor_clf.apply(X)
#    region_dict = defaultdict(list)
#    N = X.shape[0]
#    for i in np.arange(N):
#        region_dict[str(list(X_region_ids[i,:]))].append(i)
#    return X_region_ids, region_dict

#def get_aleatoric_epistemic_uncertainities_of_auditor_Regions(auditor_clf, X):
#    """Given an trained ensemble auditor, compute predicted prob of error, aleatoric and epistemic uncertainities
#    Args:
#        auditor_clf: Fitted auditor ensemble model
#        X: data records x \in X
#    Returns:
#        pred_prob_error
#        aleatoric_uncertainity: np.array of aleatoric_uncertainity for each x \in X
#        epistemic_uncertainities: np.array of aleatoric_uncertainity for each x \in X
#    """
#    est_region_p_y_x = get_all_DTree_region_p_y_x(auditor_clf)
#    
#    _, auditor_region_dict = getRegionIDs(auditor_clf, X)
#    
#    auditor_regions = np.array(list(auditor_region_dict.keys()))
#    
#    all_region_hyp_p_y_h_x = []
#    for R_i in auditor_regions:
#        R_i = ast.literal_eval(R_i)
#        t = 0
#        p_y_h_x = []
#        for R_i_t in R_i:
#            p_y_h_x.append(est_region_p_y_x[t][R_i_t])
#            t+=1
#        p_y_h_x = np.array(p_y_h_x)
#        all_region_hyp_p_y_h_x.append(p_y_h_x)
#    all_region_hyp_p_y_h_x = np.array(all_region_hyp_p_y_h_x)
#    all_region_hyp_p_y_h_x = np.transpose(all_region_hyp_p_y_h_x,(1,0,2))
#
#    p_y_x_marginal_h = np.array(all_region_hyp_p_y_h_x).mean(axis=0)
#    total_uncertainity_R = compute_total_uncertainity(p_y_x_marginal_h)
#    aleatoric_uncertainity_R = compute_aleatoric_uncertainity(all_region_hyp_p_y_h_x)
#    epistemic_uncertainity_R = total_uncertainity_R - aleatoric_uncertainity_R
#    
#    return p_y_x_marginal_h[:,1], aleatoric_uncertainity_R, epistemic_uncertainity_R, auditor_region_dict, auditor_regions
    
#def get_all_DTree_region_p_y_x(auditor_clf):
#    '''returns list of dictionary of size number of estimators(in the ensemble) 
#   e.g., [{r_0_0:p_y_x, r_0_1:p_y_x},
#          {r_1_0:p_y_x},
#          {r_2_0:p_y_x, r_2_1:p_y_x, r_2_2:p_y_x}] 
#   where r_i_t is a region "i" decision tree "t", and p_y_x is is the prob of outcomes p(y|x) in region r_i_t'''
#
#    aud_estimators = auditor_clf.estimators_
#
#    est_region_p_y_x = []
#    for i in range(len(aud_estimators)):
#        est = aud_estimators[i]
#        X_region_ids = est.apply(X)
#        p_y_x = est.predict_proba(X_test)
#        R_p_y_x = dict(zip(X_region_ids, p_y_x))
#        est_region_p_y_x.append(R_p_y_x)
#
#    return est_region_p_y_x