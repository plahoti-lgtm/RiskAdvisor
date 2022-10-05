from utils.training_helpers import train_Auditor_CV, train_Auditor
import joblib

import os, sys
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import regularizers

def model_cifar10_ResNet50(num_classes = 10):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()    
    
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255.
    X_test /= 255.
    
    X_train_xgb = X_train.reshape((X_train.shape[0], -1))
    X_test_xgb = X_test.reshape((X_test.shape[0], -1)) 
    
    # Onehot Classes
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes = num_classes)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes = num_classes)
    
    input_shape = X_train.shape[1:]
    
    ResNet50_model = tf.keras.applications.ResNet50(include_top = True,
                                                    weights = None,
                                                   input_tensor = None,
                                                   input_shape = input_shape,
                                                   pooling = max,
                                                   classes = num_classes,
                                                   classifier_activation = 'softmax')
    
    ResNet50_model.summary()
            
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    return (X_train, X_train_xgb, y_train, y_train_onehot), (X_test, X_test_xgb, y_test, y_test_onehot),  ResNet50_model


def model_mnist_CNN(num_classes = 10):
    
    input_shape = (28,28,1)
    
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255.
    X_test /= 255.
    
    X_train = np.expand_dims(X_train,-1)
    X_test = np.expand_dims(X_test,-1)
    
    X_train_xgb = X_train.reshape((X_train.shape[0], -1))
    X_test_xgb = X_test.reshape((X_test.shape[0], -1)) 
    
     # Onehot Classes
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes = num_classes)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes = num_classes)
                
    CNN_model = tf.keras.Sequential(
        [
        layers.InputLayer(input_shape),
        layers.Conv2D(32, kernel_size=(3,3),activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, kernel_size=(3,3),activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dense(num_classes, activation = "softmax")    
        ]
    )
    
    CNN_model.summary()
    
    return (X_train, X_train_xgb, y_train, y_train_onehot), (X_test, X_test_xgb, y_test, y_test_onehot), CNN_model

def model_fashion_mnist_CNN(num_classes = 10):
    
    input_shape = (28,28,1)
    
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255.
    X_test /= 255.
    
    X_train = np.expand_dims(X_train,-1)
    X_test = np.expand_dims(X_test,-1)
    
    X_train_xgb = X_train.reshape((X_train.shape[0], -1))
    X_test_xgb = X_test.reshape((X_test.shape[0], -1)) 
    
     # Onehot Classes
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes = num_classes)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes = num_classes)
                
    CNN_model = tf.keras.Sequential(
        [
        layers.InputLayer(input_shape),
        layers.Conv2D(32, kernel_size=(3,3),activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, kernel_size=(3,3),activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dense(num_classes, activation = "softmax")    
        ]
    )
    
    CNN_model.summary()
    
    return (X_train, X_train_xgb, y_train, y_train_onehot), (X_test, X_test_xgb, y_test, y_test_onehot), CNN_model

def load_dataset_and_bbox_model(dataset_name, BBox_name, trained_bbox_model_data_path = None):
    
    if trained_bbox_model_data_path:
            
            bbox_clf = tf.keras.models.load_model(os.path.join(trained_bbox_model_data_path, 'trained_bbox_model'))
            
            X_train = np.load(os.path.join(trained_bbox_model_data_path,'X_train.npy'))
            X_test = np.load(os.path.join(trained_bbox_model_data_path,'X_test.npy'))        
            
            X_train_xgb = np.load(os.path.join(trained_bbox_model_data_path,'X_train_xgb.npy'))
            X_test_xgb = np.load(os.path.join(trained_bbox_model_data_path,'X_test_xgb.npy')) 
            
            y_train_onehot = np.load(os.path.join(trained_bbox_model_data_path,'y_train_onehot.npy'))
            y_test_onehot = np.load(os.path.join(trained_bbox_model_data_path,'y_test_onehot.npy'))
            
            y_train = np.load(os.path.join(trained_bbox_model_data_path,'y_train.npy'))
            y_test = np.load(os.path.join(trained_bbox_model_data_path,'y_test.npy'))
            
            return (X_train, X_train_xgb, y_train, y_train_onehot), (X_test, X_test_xgb, y_test, y_test_onehot), bbox_clf
        
    else:
            
        if dataset_name in ['mnist','mnist_ood'] and BBox_name=='CNN':
            (X_train, X_train_xgb, y_train, y_train_onehot), (X_test, X_test_xgb, y_test, y_test_onehot), bbox_clf = model_mnist_CNN()
            
            # If ood_classes is set, excludes mentioned classes from train data
            if dataset_name =='mnist_ood':
                ood_classes = [5]        
                mask = ~np.isin(y_train, ood_classes)
            
                X_train = X_train[mask]
                X_train_xgb = X_train_xgb[mask]
                
                y_train = y_train[mask]
                y_train_onehot = y_train_onehot[mask]
            
            bbox_clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
            bbox_clf.fit(X_train,y_train_onehot,batch_size = 128, epochs = 10, validation_split = 0.1)
            
            results = bbox_clf.evaluate(X_test, y_test_onehot)
            print(results)
            
        elif dataset_name in ['f_mnist','f_mnist_ood'] and BBox_name=='CNN':
            (X_train, X_train_xgb, y_train, y_train_onehot), (X_test, X_test_xgb, y_test, y_test_onehot), bbox_clf = model_fashion_mnist_CNN()
            
            # If ood_classes is set, excludes mentioned classes from train data
            if dataset_name =='f_mnist_ood':
                ood_classes = [5]        
                mask = ~np.isin(y_train, ood_classes)
            
                X_train = X_train[mask]
                X_train_xgb = X_train_xgb[mask]
                
                y_train = y_train[mask]
                y_train_onehot = y_train_onehot[mask]
            
            bbox_clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
            bbox_clf.fit(X_train,y_train_onehot,batch_size = 128, epochs = 10, validation_split = 0.1)
            
            results = bbox_clf.evaluate(X_test, y_test_onehot)
            print(results)
                    
        elif dataset_name in ['cifar10','cifar10_ood'] and BBox_name=='ResNet50':
            (X_train, X_train_xgb, y_train, y_train_onehot), (X_test, X_test_xgb, y_test, y_test_onehot), bbox_clf = model_cifar10_ResNet50()            
                        
            # If ood_classes is set, excludes mentioned classes from train data
            if dataset_name == 'cifar10_ood':
                ood_classes = [0]        
                
                mask = ~np.isin(y_train, ood_classes)
            
                X_train = X_train[mask]
                X_train_xgb = X_train_xgb[mask]
                
                y_train = y_train[mask]
                y_train_onehot = y_train_onehot[mask]
                
            bbox_clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
            bbox_clf.fit(X_train,y_train_onehot,batch_size = 128, epochs = 100, validation_split = 0.1)
            
            results = bbox_clf.evaluate(X_test, y_test_onehot)
            print(results)            
            
        elif dataset_name in ['cifar100'] and BBox_name=='ResNet50':
            (X_train, X_train_xgb, y_train, y_train_onehot), (X_test, X_test_xgb, y_test, y_test_onehot), bbox_clf = model_cifar100_ResNet50()            
                        
            bbox_clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
            bbox_clf.fit(X_train,y_train_onehot,batch_size = 128, epochs = 150, validation_split = 0.1)
            
            results = bbox_clf.evaluate(X_test, y_test_onehot)
            print(results)            
        else:
            print('Error for {}:{}:{}'.format(dataset_name, BBox_name, trained_bbox_model_data_path))
        
        return (X_train, X_train_xgb, y_train, y_train_onehot), (X_test, X_test_xgb, y_test, y_test_onehot), bbox_clf

def train_auditor_model(X_train_xgb, y_train_errors, Auditor_name, auditor_params, gridSearchCV):
        
    if gridSearchCV:
        auditor_clf = train_Auditor_CV(X_train = X_train_xgb, 
                                error_train = y_train_errors,
                                name = Auditor_name,
                                param_grid=auditor_params,
                                scoring = scoring)
    else:
        auditor_clf = train_Auditor(X_train = X_train_xgb, 
                                error_train = y_train_errors,
                                name = Auditor_name,
                                params=auditor_params)
    
    print('Auditor Model fitted...')    
    return auditor_clf
     

if __name__ == '__main__':
    if __name__ == '__main__':
        
        dataset_name = sys.argv[1] # 'cifar10_ood', 'cifar10', 'mnist', 'mnist_ood'
        BBox_name = sys.argv[2] #'ResNet50', CNN
        TRAIN_BBOX = sys.argv[3].lower() == 'true'
        TRAIN_AUDITOR = sys.argv[4].lower() == 'true'       
        
        scoring = None            
        gridSearchCV = True
        INCLUDE_BBOX_OUTPUT = True 
        
#        BBox_name = 'ResNet50'
#        batch_size = 128
#        epochs = 100        
#        num_classes = 10  
        
#        BBox_name = 'CNN'
#        ood_classes = [5, 8]        
#        batch_size = 128
#        epochs = 10        
#        num_classes = 10        
        
        experiment_FLAG = '{}'.format(scoring)
        
        # Ensemble of Stochastic Gradient Boosted Trees - Catboost implementation
        Auditor_name, auditor_params = ('EnsembleOfCBTs', 
              {'est_names': ['cbt','cbt','cbt','cbt','cbt','cbt','cbt','cbt','cbt','cbt'],
              'est_params':[{"n_estimators":[1000], 'subsample':[0.9,0.5,0.75], 'depth':[4,6], 'random_seed':[1], 'learning_rate':[0.001, 0.01, 0.1]},
                            {"n_estimators":[1000], 'subsample':[0.9,0.5,0.75], 'depth':[4,6], 'random_seed':[2], 'learning_rate':[0.001, 0.01, 0.1]},
                            {"n_estimators":[1000], 'subsample':[0.9,0.5,0.75], 'depth':[4,6], 'random_seed':[3], 'learning_rate':[0.001, 0.01, 0.1]},
                            {"n_estimators":[1000], 'subsample':[0.9,0.5,0.75], 'depth':[4,6], 'random_seed':[4], 'learning_rate':[0.001, 0.01, 0.1]},
                            {"n_estimators":[1000], 'subsample':[0.9,0.5,0.75], 'depth':[4,6], 'random_seed':[5], 'learning_rate':[0.001, 0.01, 0.1]},
                            {"n_estimators":[1000], 'subsample':[0.9,0.5,0.75], 'depth':[4,6], 'random_seed':[6], 'learning_rate':[0.001, 0.01, 0.1]},
                            {"n_estimators":[1000], 'subsample':[0.9,0.5,0.75], 'depth':[4,6], 'random_seed':[7], 'learning_rate':[0.001, 0.01, 0.1]},
                            {"n_estimators":[1000], 'subsample':[0.9,0.5,0.75], 'depth':[4,6], 'random_seed':[8], 'learning_rate':[0.001, 0.01, 0.1]},
                            {"n_estimators":[1000], 'subsample':[0.9,0.5,0.75], 'depth':[4,6], 'random_seed':[8], 'learning_rate':[0.001, 0.01, 0.1]},
                            {"n_estimators":[1000], 'subsample':[0.9,0.5,0.75], 'depth':[4,6], 'random_seed':[10], 'learning_rate':[0.001, 0.01, 0.1]}
                            ]})

        if INCLUDE_BBOX_OUTPUT:
            experiment_FLAG += 'InclBBoxOutput'    
        
        if TRAIN_BBOX:
            trained_bbox_model_data_path = None
        else:
            trained_bbox_model_data_path = './results/{}/{}/EnsembleOfCBTs/NoneInclBBoxOutput/'.format(dataset_name, BBox_name)
        
        results_base_dir = './results/{}/'.format(dataset_name)
        if not os.path.isdir(results_base_dir):
            os.makedirs(results_base_dir)
            
        results_dir = os.path.join(results_base_dir,'{}/{}/{}'.format(BBox_name, Auditor_name, experiment_FLAG))    
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        if TRAIN_BBOX:            
            (X_train, X_train_xgb, y_train, y_train_onehot), (X_test, X_test_xgb, y_test, y_test_onehot), bbox_clf = load_dataset_and_bbox_model(dataset_name, BBox_name, trained_bbox_model_data_path)
            
            print('Dumping Trained BBoxs Models and Data at {}'.format(results_dir))   
            
            output_file_path = os.path.join(results_dir,'trained_bbox_model')        
            tf.keras.models.save_model(bbox_clf, output_file_path, overwrite=True, include_optimizer=True, save_format=None, signatures=None, options=None, save_traces=True) 
            
            y_train_pred_prob = bbox_clf.predict(X_train)
            y_test_pred_prob = bbox_clf.predict(X_test)
            
            y_train_pred = np.argmax(y_train_pred_prob,axis=1)        
            y_test_pred = np.argmax(y_test_pred_prob,axis=1)        
            
            y_train_errors = (y_train!=y_train_pred).astype(int)
            y_test_errors = (y_test!=y_test_pred).astype(int)
            
            np.save(os.path.join(results_dir,'X_train'),X_train)
            np.save(os.path.join(results_dir,'X_test'),X_test)        
            
            np.save(os.path.join(results_dir,'X_train_xgb'),X_train_xgb)
            np.save(os.path.join(results_dir,'X_test_xgb'),X_test_xgb) 
            
            np.save(os.path.join(results_dir,'y_train_onehot'),y_train_onehot)
            np.save(os.path.join(results_dir,'y_test_onehot'),y_test_onehot)
            
            np.save(os.path.join(results_dir,'y_train'),y_train)
            np.save(os.path.join(results_dir,'y_test'),y_test) 
            
            np.save(os.path.join(results_dir,'y_train_pred'),y_train_pred)
            np.save(os.path.join(results_dir,'y_test_pred'),y_test_pred)     
            
            np.save(os.path.join(results_dir,'y_train_pred_prob'),y_train_pred_prob)
            np.save(os.path.join(results_dir,'y_test_pred_prob'),y_test_pred_prob)     
             
            np.save(os.path.join(results_dir,'y_train_errors'),y_train_errors)
            np.save(os.path.join(results_dir,'y_test_errors'),y_test_errors)
        
        if TRAIN_AUDITOR:
            
            X_train_xgb = np.load(os.path.join(trained_bbox_model_data_path,'X_train_xgb.npy'))
            y_train_errors = np.load(os.path.join(trained_bbox_model_data_path,'y_train_errors.npy'))
            
            X_test_xgb = np.load(os.path.join(trained_bbox_model_data_path,'X_test_xgb.npy'))
            y_test_errors = np.load(os.path.join(trained_bbox_model_data_path,'y_test_errors.npy'))
            
            if INCLUDE_BBOX_OUTPUT:
                
                print('Loading Trained Models at {}'.format(trained_bbox_model_data_path))                    
                bbox_clf = tf.keras.models.load_model(os.path.join(trained_bbox_model_data_path, 'trained_bbox_model'))
                
                X_train = np.load(os.path.join(trained_bbox_model_data_path,'X_train.npy'))
                X_test = np.load(os.path.join(trained_bbox_model_data_path,'X_test.npy'))        
                
                X_train_model_output = bbox_clf.predict(X_train)
                X_test_model_output = bbox_clf.predict(X_test)
                
                X_train_xgb = np.concatenate([X_train_xgb, X_train_model_output],axis=1)
                X_test_xgb = np.concatenate([X_test_xgb, X_test_model_output],axis=1)
            
            print('Training Auditor')
            auditor_clf = train_auditor_model(X_train_xgb, y_train_errors, Auditor_name, auditor_params, gridSearchCV)
            
            y_train_errors_pred_prob = auditor_clf.predict_proba(X_train_xgb)[:,1]
            y_train_errors_pred = auditor_clf.predict(X_train_xgb)
            
            y_test_errors_pred_prob = auditor_clf.predict_proba(X_test_xgb)[:,1]
            y_test_errors_pred = auditor_clf.predict(X_test_xgb)          
                    
            print('Dumping Trained BBos Models and Data at {}'.format(results_dir))    
            
            output_file_path = os.path.join(results_dir,'trained_auditor_model')
            joblib.dump(auditor_clf, output_file_path)
            
            np.save(os.path.join(results_dir,'y_train_errors_pred'),y_train_errors_pred)
            np.save(os.path.join(results_dir,'y_test_errors_pred'),y_test_errors_pred)
            
            np.save(os.path.join(results_dir,'y_train_errors_pred_prob'),y_train_errors_pred_prob)
            np.save(os.path.join(results_dir,'y_test_errors_pred_prob'),y_test_errors_pred_prob)
            

