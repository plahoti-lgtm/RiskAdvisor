import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class PreprocessDataset:
    def __init__(self, dataset, dataset_base_dir):
        self.dataset = dataset
        self.dataset_base_dir = dataset_base_dir
        
        if self.dataset == 'uci_adult':
            self.TRAIN_FILE = os.path.join(self.dataset_base_dir,'male_train_female_train.csv') 
            self.TEST_FILE = os.path.join(self.dataset_base_dir,'male_test_female_test.csv')
            self.OOD_FILE = os.path.join(self.dataset_base_dir,'male_train_female_train.csv')
        
            self.columns = [
                "age", "workclass", 
                "fnlwgt", 
                "education", "education-num",
                "marital-status", "occupation", "relationship", "race", "sex",
                "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
            ]
            self.additional_flagged_columns = []
            self.continous_cols = [
                             'education-num',
                             'capital-gain',
                             'capital-loss',
                             'hours-per-week']
            self.categorical_cols = ['age',
                                     'workclass',
                                 'occupation',
                                 'native-country']
            self.binary_cols = ['race','sex']
            self.features_to_drop = []
            self.target_variable = "income"
            self.target_value = 1
            self.protected_variable = 'sex'
            self.protected_group = 1
            self.nonprotected_group = 0
            self.not_OOD_group = 0
        elif self.dataset == 'uci_adult_OOD':
            self.TRAIN_FILE = os.path.join(self.dataset_base_dir,'male_train.csv') 
            self.TEST_FILE = os.path.join(self.dataset_base_dir,'male_test_female_test.csv')
            self.OOD_FILE = os.path.join(self.dataset_base_dir,'male_train_female_train.csv')
            self.columns = [
                "age", "workclass", 
                "fnlwgt", 
                "education", "education-num",
                "marital-status", "occupation", "relationship", "race", "sex",
                "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
            ]
            self.additional_flagged_columns = []
            self.continous_cols = [
                             'education-num',
                             'capital-gain',
                             'capital-loss',
                             'hours-per-week']
            self.categorical_cols = ['age',
                                     'workclass',
                                 'occupation',
                                 'native-country']
            self.binary_cols = ['race']
            self.features_to_drop = ['sex']
            self.target_variable = "income"
            self.target_value = 1
            self.protected_variable = 'sex'
            self.protected_group = 1
            self.nonprotected_group = 0
            self.not_OOD_group = 0            
        elif self.dataset =='law_school':
            self.TRAIN_FILE = os.path.join(self.dataset_base_dir,'white_train_notwhite_train.csv') 
            self.TEST_FILE = os.path.join(self.dataset_base_dir,'white_test_notwhite_test.csv')
            self.OOD_FILE = os.path.join(self.dataset_base_dir,'white_train_notwhite_train.csv')
            
            self.columns = ['zfygpa', 'zgpa', 'DOB_yr', 'isPartTime', 'sex', 'race', 'cluster_tier', 'family_income', 'lsat', 'ugpa', 'pass_bar', 'weighted_lsat_ugpa']
            self.additional_flagged_columns = []
            
            self.continous_cols = ['zfygpa', 'zgpa',  'cluster_tier', 'family_income', 'lsat', 'ugpa', 'weighted_lsat_ugpa']
            self.categorical_cols = ['DOB_yr', 'isPartTime','sex']
            self.binary_cols = []            
            self.features_to_drop = []
            
            self.target_variable = 'pass_bar'
            self.target_value = 1
            self.protected_variable = 'race'
            self.protected_group = 1
            self.nonprotected_group = 0
            self.not_OOD_group = 0
        elif self.dataset =='law_school_OOD':
            self.TRAIN_FILE = os.path.join(self.dataset_base_dir,'white_train.csv') 
            self.TEST_FILE = os.path.join(self.dataset_base_dir,'white_test_notwhite_test.csv')
            self.OOD_FILE = os.path.join(self.dataset_base_dir,'white_train_notwhite_train.csv')
            
            self.columns = ['zfygpa', 'zgpa', 'DOB_yr', 'isPartTime', 'sex', 'race', 'cluster_tier', 'family_income', 'lsat', 'ugpa', 'pass_bar', 'weighted_lsat_ugpa']
            self.additional_flagged_columns = []            
            self.continous_cols = ['zfygpa', 'zgpa',  'cluster_tier', 'family_income', 'lsat', 'ugpa', 'weighted_lsat_ugpa']
            self.categorical_cols = ['DOB_yr', 'isPartTime','sex']
            self.binary_cols = []
            self.features_to_drop = ['race']
            
            self.target_variable = 'pass_bar'
            self.target_value = 1
            self.protected_variable = 'race'
            self.protected_group = 1
            self.nonprotected_group = 0
            self.not_OOD_group = 0
        elif self.dataset == 'compas':
            self.TRAIN_FILE = os.path.join(self.dataset_base_dir,'full_train.csv') 
            self.TEST_FILE = os.path.join(self.dataset_base_dir,'full_test_counterfactual_W_to_B.csv')
            self.OOD_FILE = os.path.join(self.dataset_base_dir,'full_train_counterfactual_W_to_B.csv')
        
            self.columns = ['juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count',
                                'c_charge_degree', 
                                'age_cat',
                                'sex', 'race',  'is_recid']
            self.additional_flagged_columns = ['isCounterFactual']
            self.continous_cols = ['juv_fel_count',
                              'juv_misd_count',
                              'juv_other_count',
                              'priors_count',
                              ]
            self.categorical_cols = ['age_cat',
                                'c_charge_degree',
                               ]
            self.binary_cols = ['race','sex']            
            self.features_to_drop = ['isCounterFactual']
            
            self.target_variable = 'is_recid'
            self.target_value = 'Yes'
            self.protected_variable = "race"
            self.protected_group = 'Black'
            self.nonprotected_group = 'White'
            self.not_OOD_group = '_dummy'
            
        elif self.dataset =='wine':
            self.TRAIN_FILE = os.path.join(self.dataset_base_dir,'white_train.csv') 
            self.TEST_FILE = os.path.join(self.dataset_base_dir,'white_test_red_test.csv')
            self.OOD_FILE = os.path.join(self.dataset_base_dir,'white_train_red_train.csv')
            
#            self.TRAIN_FILE = os.path.join(self.dataset_base_dir,'white_train.csv') 
#            self.TEST_FILE = os.path.join(self.dataset_base_dir,'white_1_red_1.0_test.csv')
#            self.OOD_FILE = os.path.join(self.dataset_base_dir,'white_red_train.csv')
            
            self.columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                       'pH', 'sulphates', 'alcohol', 'quality','isGood', 'type']
            self.additional_flagged_columns = []            
            self.continous_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                       'pH', 'sulphates', 'alcohol']
            self.categorical_cols = []
            self.binary_cols = []
            self.features_to_drop = ['quality', 'type']
            
            self.target_variable = 'isGood'
            self.target_value = 1
            self.protected_variable = 'type'
            self.protected_group = 'red'
            self.nonprotected_group = 'white'            
            self.not_OOD_group = 'white'
            
        elif self.dataset =='heart':
            self.TRAIN_FILE = os.path.join(self.dataset_base_dir,'US_train.csv') 
            self.TEST_FILE = os.path.join(self.dataset_base_dir,'US_test_switzerland_test_hungarian_test_UK_test.csv')
            self.OOD_FILE = os.path.join(self.dataset_base_dir,'US_train_switzerland_train_hungarian_train_UK_train.csv')
            
            self.columns = ["age","sex","cp","restbps","chol","fbs","restecg","max_heart_rate","exang","oldpeak","slope","ca","num_of_vessels","diagnosis","region"]
            self.additional_flagged_columns = []            
            self.continous_cols = ["age","cp","restbps","chol","fbs","restecg","max_heart_rate","exang","oldpeak","slope","ca","num_of_vessels"]
            self.categorical_cols = []
            self.binary_cols = ['sex']            
            self.features_to_drop = ['region']
            
            self.target_variable = 'diagnosis'
            self.target_value = 1
            self.protected_variable = 'region'
            self.protected_group = 'not-US'
            self.nonprotected_group = 'US'
            self.not_OOD_group = 'US'
        
        else:
            print('Dataset:{} not Implemented'.format(dataset))
        
    def loadData(self, flagNoisy=False):
        '''load dataset'''
        self.flagNoisy = flagNoisy
        
        with open(self.TRAIN_FILE, "r") as TRAIN_FILE:
            self.train_df = pd.read_csv(TRAIN_FILE,sep=',',usecols=self.columns)
            self.train_df, (self.X_train,self.y_train), self.continuous_Pipeline, self.categorical_Pipeline = self.processdf(self.train_df,
                                                                                  dataset = self.dataset,
                                                                                  features_to_drop = self.features_to_drop, 
                                                                                  target_variable = self.target_variable, 
                                                                                  continous_cols = self.continous_cols,
                                                                                  binary_cols = self.binary_cols,
                                                                                  categorical_cols = self.categorical_cols,
                                                                                  continuous_Pipeline = None,
                                                                                  categorical_Pipeline = None,
                                                                                  mode='TRAIN')
    
        with open(self.TEST_FILE, "r") as TEST_FILE:
            self.test_df = pd.read_csv(TEST_FILE,sep=',',usecols=self.columns+self.additional_flagged_columns)
            self.test_df, (self.X_test, self.y_test), _ , _ = self.processdf(self.test_df, 
                                                         dataset = self.dataset,
                                                         features_to_drop = self.features_to_drop,
                                                         target_variable = self.target_variable,
                                                         continous_cols = self.continous_cols,
                                                         categorical_cols = self.categorical_cols,
                                                         binary_cols = self.binary_cols,
                                                         continuous_Pipeline = self.continuous_Pipeline,
                                                         categorical_Pipeline = self.categorical_Pipeline)
        
        with open(self.OOD_FILE, "r") as OOD_FILE:
            self.ood_df = pd.read_csv(OOD_FILE,sep=',',usecols=self.columns+self.additional_flagged_columns)
            self.ood_df, (self.X_ood, self.y_ood), _, _= self.processdf(self.ood_df,  
                                                    dataset = self.dataset,
                                                    features_to_drop = self.features_to_drop,
                                                    target_variable = self.target_variable,
                                                    continous_cols = self.continous_cols,
                                                    categorical_cols = self.categorical_cols,
                                                    binary_cols = self.binary_cols,
                                                    continuous_Pipeline = self.continuous_Pipeline,
                                                    categorical_Pipeline = self.categorical_Pipeline)
        if self.categorical_cols:
            self.categorical_cols_onehot = list(self.categorical_Pipeline['one_hot'].get_feature_names())
            self.feature_names_after_encoding = self.continous_cols + self.categorical_cols_onehot + self.binary_cols
        else:
            self.feature_names_after_encoding = self.continous_cols + self.categorical_cols + self.binary_cols
        
        self.test_group_membership = np.array(self.test_df[self.protected_variable])
        self.test_sensitive_ixs = np.where(self.test_group_membership!= self.nonprotected_group)[0]
        self.test_nonsensitive_ixs = np.where(self.test_group_membership== self.nonprotected_group)[0]
        self.isOOD = np.array(self.test_group_membership!=self.not_OOD_group).astype(int)
        
        if self.flagNoisy:
            self.isNoisy = np.array(self.test_df['isNoisy'])
    
    def convert_object_type_to_category(self, df):
        """Converts columns of type object to category."""
        df = pd.concat([df.select_dtypes(include=[], exclude=['object']),
                      df.select_dtypes(['object']).apply(pd.Series.astype, dtype='category')
                      ], axis=1).reindex(df.columns, axis=1)
        return df
    
    def fixAdultDataIssues(self, df):
#        dropped = df.dropna()
#        count = df.shape[0] - dropped.shape[0]
#        
#        if count > 0:
#            print("Missing Data: {} rows removed.".format(count))
#        df = dropped
        
        df = self.convert_object_type_to_category(df)
        
        df = df.replace(to_replace=['>50K','<=50K'],value=[1, 0])
        
        # Hack to Fix data issues in the dataset 
        df = df.replace(to_replace=['Male', 'Female'],value=[0, 1])
        df = df.replace(to_replace=['White',
                                    'Black',
                                    'Asian-Pac-Islander',
                                    'Amer-Indian-Eskimo',
                                    'Other'],
                        value=[0,1, 0, 0, 0])
        df['native-country'] = df['native-country'].apply(lambda x: x if x ==' United-States' else 'Other')
        
        df['age'] = pd.qcut(df['age'], q=3, labels=['q1','q2','q3'])
    
        df['capital-gain'] = df['capital-gain'].apply(lambda x: 1 if x>0 else 0)
        df['capital-loss'] = df['capital-loss'].apply(lambda x: 1 if x>0 else 0)
        
        # Combine Government Jobs
        df = df.replace(to_replace=[' State-gov',' Federal-gov',' Local-gov'],
                        value=['Government','Government','Government'])
        
        # Combine Self-Employed Jobs
        df = df.replace(to_replace=[' Self-emp-not-inc',' Self-emp-inc'], value=['Self-employed','Self-employed'])
        
        # Combine Service Jobs
        df = df.replace(to_replace=[' Other-service',
                                    ' Protective-serv',
                                    ' Armed-Forces',
                                    ' Priv-house-serv',
                                    ' Tech-support'],
                        value = ['Service','Service','Service','Service','Service'])
        # Combine Blue Collar Jobs
        df = df.replace(to_replace=[' Handlers-cleaners',
                                    ' Craft-repair',
                                    ' Transport-moving',
                                    ' Farming-fishing',
                                    ' Machine-op-inspct'],
                        value = ['Blue-collar','Blue-collar','Blue-collar','Blue-collar','Blue-collar'])
        return df
    
    def fixCompasDataIssues(self, df):
    
#        dropped = df.dropna()
#        count = df.shape[0] - dropped.shape[0]
#        
#        if count > 0:
#            print("Missing Data: {} rows removed.".format(count))
#        df = dropped
    
        df = self.convert_object_type_to_category(df)
        df['isMissing'] = df.isna().any(axis=1).astype(int)
        
        df = df.replace(to_replace=['Male', 'Female'],value=[0, 1])
        df = df.replace(to_replace=['Caucasian',
                                    'African-American',
                                    'Hispanic',
                                    'Native American',
                                    'Asian',
                                    'Other'],
                        value=[0, 1, 0, 0, 0, 0])
        
        return df
    
    def fixWineDataIssues(self, df):
    
#        dropped = df.dropna()
#        count = df.shape[0] - dropped.shape[0]
#        
#        if count > 0:
#            print("Missing Data: {} rows removed.".format(count))
#        df = dropped
        
        df = self.convert_object_type_to_category(df)
        
        if self.flagNoisy:
            missing = np.array(df.isna().any(axis=1).astype(int))
            noisy = np.array(df.quality.isin([5]).astype(int))
            df['isNoisy'] = np.array(np.bitwise_or(missing,noisy)).astype(int)
        
        #df['isGood'] = df.quality.apply(lambda x: 1 if x>=6 else 0)
        
        return df
    
    def fixLawSchoolDataIssues(self, df):
    
#        dropped = df.dropna()
#        count = df.shape[0] - dropped.shape[0]
#        
#        if count > 0:
#            print("Missing Data: {} rows removed.".format(count))
#        df = dropped
    
        df = self.convert_object_type_to_category(df)
        df = df.replace(to_replace=['Failed_or_not_attempted', 'Passed'],value=[1, 0])
        
        df = df.replace(to_replace=['Male', 'Female'],value=[0, 1])
        df = df.replace(to_replace=['White', 'Not-White'],value=[0, 1])
        
        df['DOB_yr'] = pd.qcut(df['DOB_yr'], q=3, labels=['q1','q2','q3'])
        
        return df
    
    def fixHeartDataIssues(self, df):
    
#        dropped = df.dropna()
#        count = df.shape[0] - dropped.shape[0]
#        
#        if count > 0:
#            print("Missing Data: {} rows removed.".format(count))
#        df = dropped
    
        df = self.convert_object_type_to_category(df)
        
        if self.flagNoisy:
            df['isNoisy'] = df.isna().any(axis=1).astype(int)
        
#        df['age'] = pd.qcut(df['age'], q=10, labels = np.arange(1,11,1))
        
        return df
    
    def processdf(self, df, dataset, features_to_drop, target_variable, continous_cols, categorical_cols, binary_cols, continuous_Pipeline, categorical_Pipeline, mode='TEST'):
        print('Processing {} Dataset'.format(dataset))
        
        if dataset in ['compas']:
            df = self.fixCompasDataIssues(df)
        elif dataset in ['uci_adult', 'uci_adult_OOD', 'uci_adult_with_noise']:
            df = self.fixAdultDataIssues(df)
        elif dataset in ['wine','wine_multiclass']:
            df = self.fixWineDataIssues(df)
        elif dataset in ['law_school', 'law_school_OOD', 'law_school_with_noise']:
            df = self.fixLawSchoolDataIssues(df)
        elif dataset in ['heart']:
            df = self.fixHeartDataIssues(df)
        else:
            print('Dataset:{} not Implemented'.format(dataset))
            return -1
        
        #drops target variable and other features not to be kept in X
        features_to_keep = list(set(df.columns.tolist()) - set(features_to_drop) - set([target_variable]))
        X_df = df[features_to_keep] 
        
        X_continous = np.array(X_df[continous_cols])
        X_categorical = np.array(X_df[categorical_cols])
        X_binary = np.array(X_df[binary_cols]) 
    
        if mode=='TRAIN':

            if continous_cols:            
                continuous_Pipeline = Pipeline( [("imputer", SimpleImputer(strategy="median")),
                                             ("scaler", StandardScaler())
                                            ]
                                           )
                X_continous = continuous_Pipeline.fit_transform(X_continous)
            
            if categorical_cols:  
                categorical_Pipeline = Pipeline([("one_hot", OneHotEncoder(handle_unknown ='ignore', sparse=False))])
                X_categorical = categorical_Pipeline.fit_transform(X_categorical)
            
        else:
            
            if continous_cols:
                X_continous = continuous_Pipeline.transform(X_continous)
            
            if categorical_cols:
                X_categorical = categorical_Pipeline.transform(X_categorical)
        
        X = np.concatenate((X_continous, X_categorical, X_binary),axis=1)
        y = np.array(df[target_variable]).ravel()
    
        return df, (X, y), continuous_Pipeline, categorical_Pipeline
