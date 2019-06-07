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
import json

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
        data = np.array(data)
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


def k_cluster(data, k = 2, func = None, save = None, n_init = 10):
    """If using own clustering function, function must have a 
    cluster_centers_ and labels_ attributes"""
    data = np.array(data)
    clustering = None
    centers = None
    clustered_points = None
    
    if func is None:
        clustering = KMeans(n_clusters=k, random_state=0, n_init = n_init).fit(data)
    else:
        clustering = func(data)
        
    centers = clustering.cluster_centers_
    clusters = clustering.labels_

    if save != None:
        np.save(save + "centers", centers)
        np.save(save + "clusters", clusters)
        
    return {"centers": centers, "clusters": clusters}

def feature_discrepancy(df, cluster_info, f, num_bins = None, weight_bins = False):
    """Finds discrepancy of a feature between all data and a specific cluster by summing the 
    percentages of difference across all values in the feature
    
    df = dataframe
    
    cluster_info = tuple of cluster column name and value to cluster
    
    num_bins = force the data to a certain number of bins (recommended for continuous data)
    
    weight_bins = weight the discrepancy of each value proportionally to the number of possible values 
           so the discrepancy in each value will be worth less if the number of possible values is larger (e.g.,
           discrepancy in a cluster with 5 possible values worth more cluster with 100 possible values)"""
    #find optimal binning later
    
    feat = pd.DataFrame(df[f])[f]
    cluster = pd.DataFrame(df[df[cluster_info[0]] == cluster_info[1]][f])[f]
    
    feature_map = {}
    if num_bins != None:
        num_bins = min(len(set(df[f])), num_bins)
        bins = k_cluster([[x] for x in feat], k=num_bins)
        for item in range(len(feat)):
            feature_map[feat[item]] = bins[item]
    else:
        for item in range(len(feat)):
            feature_map[feat[item]] = feat[item]
       
    discrepancy = 0
    
    cluster_dist, feat_dist = defaultdict(int), defaultdict(int)
    for item in feature_map:
        feat_dist[feature_map[item]] = 0
        cluster_dist[feature_map[item]] = 0
        
    for item in feat:
        feat_dist[feature_map[item]] += 1

    for item in cluster:
        cluster_dist[feature_map[item]] += 1
        
    for item in feat_dist:
        feat_dist[item] = float(feat_dist[item])/len(feat)

    for item in cluster_dist:
        cluster_dist[item] = float(cluster_dist[item])/len(cluster)
    
    
    dif_arr = [abs(feat_dist[item] - cluster_dist[item]) for item in feat_dist]
    
    weighted_bins = float(1)
    if weight_bins:
        weighted_bins = float(len(dif_arr)) 
        
    for item in dif_arr:
        discrepancy += item/weighted_bins
        
    return discrepancy

def order_feature_discrepancy(df, cluster_info, feature_info, max_bins = 30, weight_bins = False):
    
    discrepancy_dict = defaultdict(list)
    for clust in cluster_info[1:]:
        for f in feature_info:
            discrepancy_dict[clust].append((f, feature_discrepancy(df, (cluster_info[0], clust), 
                                                              f, weight_bins = weight_bins, num_bins = min(len(set(df[f])), max_bins))))
            
            discrepancy_dict[clust] = sorted(discrepancy_dict[clust], key = lambda x: x[1])
                                          
    return discrepancy_dict

def display_feature_discrepancy(all_ordered):

    for item in all_ordered:
        ordered = []
        for feature in range(len(all_ordered[item])):
            ordered.append(all_ordered[item][feature][0])
        print("Ecid: {}, ordered: {}".format(item, ordered))
        print()


def prediction_discrepancy(df, cluster_info, f):
    feat = pd.DataFrame(df[f])[f]
    cluster = pd.DataFrame(df[df[cluster_info[0]] == cluster_info[1]][f])[f]
    
    feat_avg = float(np.mean(feat))
    cluster_avg = float(np.mean(cluster))
    
    return {"cluster": cluster_avg, "feat": feat_avg, "discrepancy": abs(feat_avg - cluster_avg)}

def order_prediction_discrepancy(df, cluster_info, f):
    
    discrepancy_dict = defaultdict(int)
    
    for clust in cluster_info[1:]:
        discrepancy_dict[clust] = prediction_discrepancy(df, (cluster_info[0], clust), f)                     
    return discrepancy_dict

def display_pred_accur_discrepancy(all_ordered):
    for item in all_ordered:
        print("Ecid: {}, difference: {}".format(item, all_ordered[item]))


def accuracy_discrepancy(df, cluster_info, p, l):
    feat_p = list(pd.DataFrame(df[p])[p])
    feat_l = list(pd.DataFrame(df[l])[l])
    
    cluster_p = list(pd.DataFrame(df[df[cluster_info[0]] == cluster_info[1]][p])[p])
    cluster_l = list(pd.DataFrame(df[df[cluster_info[0]] == cluster_info[1]][l])[l])
    
    feat_avg = float(np.mean([0 if feat_p[x] != feat_l[x] else 1 for x in range(len(feat_p))]))
    cluster_avg = float(np.mean([0 if cluster_p[x] != cluster_l[x] else 1 for x in range(len(cluster_p))]))
    
    return {"cluster": cluster_avg, "whole": feat_avg, "discrepancy": abs(feat_avg - cluster_avg)}


def order_accuracy_discrepancy(df, cluster_info, p, l):
    
    discrepancy_dict = defaultdict(int)
    
    for clust in cluster_info[1:]:
        discrepancy_dict[clust] = accuracy_discrepancy(df, (cluster_info[0], clust), p, l)                     
    return discrepancy_dict


def standardize(column):
    return stats.zscore(column)


def standardize_data(data, save = None):
    data = np.array(data)
    for column in range(len(data[0,:])):
        data[:,column] = standardize(data[:,column]) 

    if save != None:
        np.save(save, data) 

    return data

def matrix_from_dataframe(df, row, col):
    matrix = np.array([[0 for x in range(len(set(df[col])))] for x in range(len(set(df[row])))])
    for index, item in df.iterrows():
        x = int(item[row])
        y = int(item[col])
        matrix[x][y] += 1
        
    return matrix

    
def create_heatmap(matrix, xlabel = "x-axis", ylabel = "y-axis", title = "HeatMap", save = False):
    "given a 2D array, return a heat map (outer is x-axis, inner is y-axis)"
    plt.xlabel(xlabel, fontsize = 12)
    plt.ylabel(ylabel, fontsize = 12)
    
    plt.imshow(matrix.T, origin = "lower")
    plt.colorbar()
    
    if save:
        plt.savefig(title)

def plot_distribution(cluster, cluster_size, save = False, plt_info = []):
    
    plt.hist(cluster, bins = np.arange(cluster_size+1)-0.5)
    
    if plt_info:
        plt.xlabel(plt_info[0], fontsize = 12)
        plt.ylabel(plt_info[1], fontsize = 12)
        plt.title(plt_info[2])
    
    
    if save:
        plt.savefig(plt_info[3])
        
    plt.show()

def save_json(data, name):
    "Save dictionaries and defaultdicts"
    json.dump(data, open('{}.json'.format(name), 'w'), indent = 2)
    
def load_json(name):
    "loads dictionaries and defeaultdicts"
    return json.load(open(name, 'r'))




