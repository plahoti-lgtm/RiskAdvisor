from utils.sampling_helpers import run_sampling_experiment
from multiprocessing import Pool

# EXPERIMENT-SPECIFIC CONFIG
dataset_name = 'wine'
BBox_names = ['LogisticRegression','Neural Net','RandomForest','SVM']
Auditor_names = ['EnsembleOfGBTs','RandomForest']
_max_region_size = [None, 0.01, 0.05, 0.1] #[None, 10, 20, 30, 40]
sampling_techniques = ['random','bbox_confidence','trust_score','epistemic','baseline'] 

scoring_s = [None]
gridSearchCV = True

num_runs = 10
num_loops = 20
sampling_percentage = 2


def do(args):    
    dataset_name, BBox_name, Auditor_name, experiment_FLAG, sampling_FLAG, gridSearchCV, max_region_size, sampling_techniques, num_runs, num_loops, sampling_percentage = args

    run_sampling_experiment(dataset_name = dataset_name,
                   BBox_name = BBox_name,
                   Auditor_name = Auditor_name,
                   experiment_FLAG = experiment_FLAG,
                   sampling_FLAG = sampling_FLAG,
                   gridSearchCV = gridSearchCV,
                   max_region_size = max_region_size,
                   sampling_techniques = sampling_techniques,
                   num_runs = num_runs,
                   num_loops = num_loops,
                   sampling_percentage = sampling_percentage
                   )  
    
if __name__ == '__main__':
    if __name__ == '__main__':
        pool = Pool(processes=16)
        args_queue = []
        for BBox_name in BBox_names:
            for Auditor_name in Auditor_names:
                for scoring in scoring_s:
                    experiment_FLAG = '{}'.format(scoring)
                    for max_region_size in _max_region_size:
                        sampling_FLAG = '{}'.format(max_region_size)
                        args_queue.append((dataset_name, BBox_name, Auditor_name, experiment_FLAG, 
                                           sampling_FLAG,
                                           gridSearchCV,
                                           max_region_size, sampling_techniques,
                                           num_runs, num_loops, sampling_percentage))
        pool.map(do, args_queue)
