import numpy as np
from scipy.stats import entropy

from TrustScore import trustscore

def compute_uncertainity(p_y_x):
    """computes uncertainitity (-2*shannon_entropy) given p(y|h,x)
    Args:
        pred_prob: np.ndarray of p(y|h,x) 
                   for all x \in X
                           y \in Y
    Returns:
        a 1D array of per item uncertainities:  
    """
    return np.apply_along_axis(entropy, axis=1, arr=p_y_x, base=2)

def compute_total_uncertainity(p_y_x_marginal_h):
    """computes total uncertainitity given marginal p(y|x) over all h \in H.
    
    Args:
        pred_prob: np.ndarray of p(y|x) 
    Returns:
        a 1D array of per item uncertainities:  
    """
    return compute_uncertainity(p_y_x_marginal_h)
    
def compute_aleatoric_uncertainity(all_hyp_p_y_h_x):
    """computes total uncertainitity given p(y|h,x) for all h \in H.
    
    Args:
        pred_prob: np.ndarray of p(y|h,x) 
            for all x \in X 
                    h \in H
                    y in Y
    Returns:
        a 1D array of per item uncertainities:  
    """
    all_hyp_uncertainity = []
    for p_y_h_x in all_hyp_p_y_h_x:
        all_hyp_uncertainity.append(compute_uncertainity(p_y_h_x))
    return np.array(all_hyp_uncertainity).mean(axis=0)

def get_aleatoric_epistemic_uncertainities(ensemble_clf, X):
    """Given an trained ensemble auditor, compute predicted prob of error, aleatoric and epistemic uncertainities
    Args:
        ensemble_clf: Fitted ensemble model
        X: data records x \in X
    Returns:
        pred_prob_error
        aleatoric_uncertainity: np.array of aleatoric_uncertainity for each x \in X
        epistemic_uncertainities: np.array of aleatoric_uncertainity for each x \in X
    """
    # Fetch pred_prob P(z|h,x) of each SGBT in the E-SGBT (auditor ensemble model)
    aud_estimators = ensemble_clf.estimators_
    all_hyp_p_y_h_x = []
    for aud_est in aud_estimators:
        all_hyp_p_y_h_x.append(aud_est.predict_proba(X))
    all_hyp_p_y_h_x = np.array(all_hyp_p_y_h_x)
    
    # Compute various uncertainities given p_z_h_x for all h \in H
    p_y_given_x = np.array(all_hyp_p_y_h_x).mean(axis=0)
    total_uncertainity = compute_total_uncertainity(p_y_given_x)
    aleatoric_uncertainity = compute_aleatoric_uncertainity(all_hyp_p_y_h_x)
    epistemic_uncertainity = total_uncertainity - aleatoric_uncertainity
    
    return p_y_given_x[:,1], aleatoric_uncertainity, epistemic_uncertainity

def computeTrustScore(X_train, y_train, X_test, y_test_pred, k=10, alpha=0.01, filtering = "density"):
    '''
    Returns Trust Score on evaluation data
    '''
    # Fit Trust Score model to training data and labels
    #trust_model = trustscore.TrustScore()
    trust_model = trustscore.TrustScore(k=k, alpha=alpha, filtering=filtering)
    trust_model.fit(X_train, y_train)

    ## Compute trusts score, given (unlabeled) testing examples and bbox models predictions.
    #y_pred_test = bbox_clf.predict(X_test)
    
    trust_scores = trust_model.get_score(X_test, y_test_pred)
    trust_scores = trust_scores.astype(float)
    return trust_scores

def computeConfidence(bbox_clf, X_test):
    return bbox_clf.predict_proba(X_test).max(axis=1)