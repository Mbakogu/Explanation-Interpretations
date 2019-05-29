import sklearn
import sklearn.datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
from collections import defaultdict
import pandas as pd
from scipy import stats

class Trainer():
    
    def __init__(self, data, labels, feature_names = None, classifier = sklearn.ensemble.RandomForestClassifier(n_estimators=500)):
        
        data = np.array(data)

        labels = np.array(labels)

        self.classifier = classifier
        
        self.unchanged_data = np.array(data)
        
        self.unchanged_labels = np.array(labels)
        
        self.data = data
        
        self.labels = labels
        
        self.class_names = set(self.labels)

        self.feature_names = feature_names

        if self.feature_names == None:
            self.feature_names = [str(x) for x in range(len(self.data[0,:]))]
        
        self.categorical_names = {}
        
        self.categorical_features = []
        
        for i in range(len(self.data[0,:])):
            if self.is_categorical(self.data[:,i]):
                self.categorical_features.append(i)
                self.data[:, i], self.categorical_names[i] = self.fix_data(self.data[:, i])
         
        
        if self.is_categorical(self.labels):
            self.labels, self.class_names = self.fix_data(self.labels)
            
        self.data = self.data.astype(float)
        
        self.encoder = sklearn.preprocessing.OneHotEncoder(categorical_features=self.categorical_features)
                  
        self.train, self.test, self.labels_train, self.labels_test = sklearn.model_selection.train_test_split(self.data, self.labels, 
                                                                        train_size=0.8, test_size=0.2)
        
        self.encoder.fit(self.data)
        
        encoded_train = self.encoder.transform(self.train)
        
        self.classifier.fit(encoded_train, self.labels_train)
            
        
    def is_categorical(self, column):
        try:
            column.astype(float)
            return False
        except:
            return True
        
    def fix_data(self, data):
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(data)
        return le.transform(data), le.classes_
        
        
    def predict(self, instance):
        return self.classifier.predict_proba(self.encoder.transform(instance)).astype(float)
    
    def del_predict(self, instance):
        "not working currently"
        instance = np.array([instance])
        for i in range(len(instance[0,:])):
            if self.is_categorical(instance[:,i]):
                instance[:,i], garbage = self.fix_data(instance[:,i])
        return (self.classifier.predict_proba(self.encoder.transform((instance[0]).reshape(1,-1))).astype(float))[0,1]
    
    def get_prediction(self, instance):
        return round((self.classifier.predict_proba(self.encoder.transform((instance).reshape(1,-1))).astype(float))[0,1])

    def save_data(save = "Instances"):
        np.save(save, self.get_data())
    
    def get_data(self):
        return self.data
    
    def get_labels(self):
        return self.labels
        
    def get_categorical_features(self):
        return self.categorical_features

    def get_categorical_names(self):
        return self.categorical_names

    def get_training_data(self):
        return self.train

    def get_test_data(self):
        return self.test

    def get_training_labels(self):
        return self.labels_train

    def get_test_labels(self):
        return self.labels_test

    def get_class_names(self):
        return self.class_names

    def get_feature_names(self):
        return self.feature_names

    def get_unchanged_data(self):
        return self.unchanged_data

    def get_unchanged_labels(self):
        return self.unchanged_labels


class LimeExplainer():
    
    def __init__(self, data, labels, classifier = None):
        
        data = np.array(data)

        labels = np.array(labels)

        self.classifier = classifier
        if self.classifier == None:
            self.classifier = Trainer(data, labels)
            
        self.data = self.classifier.get_data()
            
        self.labels = self.classifier.get_labels()
            
        self.explainer = lime.lime_tabular.LimeTabularExplainer(self.classifier.get_training_data(), feature_names = self.classifier.get_feature_names(), class_names = self.classifier.get_class_names(),
                                                   categorical_features = self.classifier.get_categorical_features(), 
                                                   categorical_names = self.classifier.get_categorical_names(), kernel_width=3)
        
    def explain(self, instance, num_features = None):
        if num_features == None:
            num_features = len(self.classifier.get_feature_names())
        return self.explainer.explain_instance(instance, self.classifier.predict, num_features=num_features).as_list()

    def explain_data(self, data, save = None):
        data = np.array(data)
        explanations = []
        
        for item in range(len(data)):
            explanations.append(self.explain(data[item]))
            
        for item in range(len(explanations)):
            for ele in range(len(explanations[item])):
                explanations[item][ele] = float(explanations[item][ele][1])
                
        explanations = np.array(explanations)
                
        if save != None:
            np.save(save, explanations)
            
        return explanations
    
    def get_data(self):
        return self.data
        
    def get_labels(self):
        return self.labels


def calc_discrepancy(df, ecid, f, continuous = False, max_bins = 30):
    discrepancy = 0
    spec = df[df["ecid"] == ecid][f]
    feat = df[f]
    
    spec_dist = defaultdict(int)
    
    feat_dist = defaultdict(int)
    
    for item in spec:
        spec_dist[item] += 1

    for item in feat:
        feat_dist[item] += 1
        
    if set(spec_dist) != set(feat_dist):
        raise Exception("One of the arrays: {} or {} is missing 1 or more categories\n {}\n{}".format(ecid, f, set(spec_dist), set(feat_dist)))

    for item in spec_dist:
        spec_dist[item] = float(spec_dist[item])/len(spec)

    for item in feat_dist:
        feat_dist[item] = float(feat_dist[item])/len(feat)
    
    
    dif_arr = [abs(feat_dist[item] - spec_dist[item]) for item in feat_dist]
    
    for item in dif_arr:
        discrepancy += item/float(len(dif_arr)) 
        
    return discrepancy


def k_cluster(data, k = 2, func = None, save = None):
    data = np.array(data)
    print("Kmeans for {} clusters".format(k))
    clusters = None
    if func is None:
        clusters = KMeans(n_clusters=k, random_state=0).fit_predict(data)
    else:
        clusters = func(data)

    if save != None:
        np.save(save, clusters)

    return clusters


def standardize(column):
    return stats.zscore(column)


def standardize_data(data, save = None):
    data = np.array(data)
    for column in range(len(data[0,:])):
        data[:,column] = standardize(data[:,column]) 

    if save != None:
        np.save(save, data) 

    return data

def create_heatmap(matrix, xlabel = "x-axis", ylabel = "y-axis", title = "HeatMap", save = False):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.imshow(matrix)
    
    if save:
        plt.savefig(title)




