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
        
        self.original_data = np.array(data)
        
        self.original_labels = np.array(labels)
        
        self.data = data
        
        self.labels = labels
        
        self.class_names = set(self.labels)

        self.feature_names = feature_names

        if self.feature_names == None:
            self.feature_names = [str(x) for x in range(len(self.data[0,:]))]
        
        self.categorical_names = {}
        
        self.categorical_features = []
        
        #self.stored_categorical_features = []
        
        self.original_train, self.original_test, self.original_labels_train, self.original_labels_test = sklearn.model_selection.train_test_split(self.data, self.labels, 
                                                                        train_size=0.8, test_size=0.2)
        
        self.train = np.array(self.original_train)
        self.test = np.array(self.original_test)
        self.labels_train = np.array(self.original_labels_train)
        self.labels_test = np.array(self.original_labels_test)
        
        for i in range(len(self.data[0,:])):
            if self.is_categorical(self.data[:,i]):
                self.categorical_features.append(i)
                fixed = self.fix_data(self.data[:, i])
                self.data[:, i] = fixed[0] 
                self.categorical_names[i] = fixed[1]
                self.train[:, i] = self.fix_data(self.train[:, i])[0]
                self.test[:, i] = self.fix_data(self.test[:, i])[0]
                
        
        if self.is_categorical(self.labels):
            self.labels, self.class_names, self.original_labels = self.fix_data(self.labels)
            self.labels_train = self.fix_data(self.labels_train)[0]
            self.labels_test = self.fix_data(self.labels_test)[0]
            
        self.data = self.data.astype(float)
        self.test = self.test.astype(float)
        self.train = self.train.astype(float)
        
        self.encoder = sklearn.preprocessing.OneHotEncoder(categorical_features=self.categorical_features)
        
        self.encoder.fit(self.data)
        
        encoded_train = self.encoder.transform(self.train)
        
        self.classifier.fit(encoded_train, self.labels_train)
            
        
    def is_categorical(self, column):
        column = np.array(column) #recently added
        try:
            column.astype(float)
            return False
        except:
            return True
        
    def fix_data(self, data):
        data = np.array(data)
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(data)
        
        classes = le.classes_
        transformed = le.transform(data)
        original = le.inverse_transform(np.array(transformed))
        
        return transformed, classes, original
        
        
    def predict(self, instance):
        return self.classifier.predict_proba(self.encoder.transform(instance)).astype(float)
    
    def del_predict(self, instance):
        "not working currently"
        instance = np.array([instance])
        for i in range(len(instance[0,:])):
            if self.is_categorical(instance[:,i]):
                instance[:,i], garbage1, garbage2 = self.fix_data(instance[:,i])
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
    
    def get_class_names(self):
        return self.class_names

    def get_feature_names(self):
        return self.feature_names

    def get_training_data(self, save = None):
        if save != None:
            np.save(save, self.train)
            
        return self.train

    def get_test_data(self, save = None):
        if save != None:
            np.save(save, self.test)
            
        return self.test

    def get_training_labels(self, save = None):
        if save != None:
            np.save(save, self.labels_train)
            
        return self.labels_train

    def get_test_labels(self, save = None):
        if save != None:
            np.save(save, self.labels_test)
            
        return self.labels_test

    def get_original_data(self, save = None):
        if save != None:
            np.save(save, self.original_data)
            
        return self.original_data

    def get_original_labels(self, save = None):
        if save != None:
            np.save(save, self.original_labels)
            
        return self.original_labels
    
    def get_original_train_labels(self, save = None):
        if save != None:
            np.save(save, self.original_labels_train)
            
        return self.original_labels_train
        
    def get_original_test_labels(self, save = None):
        if save != None:
            np.save(save, self.original_labels_test)
            
        return self.original_labels_test
    
    def get_original_test_data(self, save = None):
        if save != None:
            np.save(save, self.original_test)
            
        return self.original_test
        
    def get_original_train_data(self, save = None):
        if save != None:
            np.save(save, self.original_train)
            
        return self.original_train


class LimeExplainer():
    
    def __init__(self, data = None, labels = None, classifier = None):

        self.classifier = classifier
        self.data = data
        self.labels = labels

        if self.classifier == None:
            self.classifier = Trainer(np.array(data), np.array(labels))
        else:
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

def calculate_discrepancy(df, cluster_info, pred_info, accur_info, feature_info, max_bins = 30, save = None):
    
    prediction_d = order_prediction_discrepancy(df, cluster_info, pred_info)
    
    accuracy_d = order_accuracy_discrepancy(df, cluster_info, accur_info[0], accur_info[1])

    feature_d = order_feature_discrepancy(df, cluster_info, feature_info, max_bins = max_bins)
    for item in feature_d:
        for ele in range(len(feature_d[item])):
            feature_d[item][ele] = (feature_d[item][ele][0].replace("_original", ""), feature_d[item][ele][1], feature_d[item][ele][2])

      
    disc = {}
    global_points = len(df)
    avg_prediction = prediction_d[0]["overall"]
    avg_accuracy = accuracy_d[0]["overall"]
    disc["global"] = {"num_points": global_points, "avg_pred": avg_prediction, "avg_acc": avg_accuracy}
    for clust_id in feature_d:
        clust_info = {}
        clust_info["num_points"] = len(df[df[cluster_info[0]] == clust_id])
        clust_info["avg_pred"] = prediction_d[clust_id]["cluster"]
        clust_info["avg_acc"] = accuracy_d[clust_id]["cluster"]
        clust_info["pred_disc"] = prediction_d[clust_id]["discrepancy"]
        clust_info["acc_disc"] = accuracy_d[clust_id]["discrepancy"]

        feature_hold = []
        for feature_information in range(len(feature_d[clust_id])):
            feat_dict = {}
            feature_name = feature_d[clust_id][feature_information][0]
            feature_discrep = feature_d[clust_id][feature_information][1]
            feature_top_k = feature_d[clust_id][feature_information][2]

            discrep_info = {}
            discrep_info["discrepancy"] = feature_discrep

            discrep_vals = {}
            for val in feature_top_k:
                discrep_vals[val[0]] = {"count": val[2], "global": val[3], "cluster": val[4], "discrepancy": val[1]}

            discrep_info["values"] = discrep_vals

            feat_dict[feature_name] = discrep_info

            feature_hold.append(feat_dict)

        clust_info["features"] = feature_hold

        disc["cluster_" + clust_id] = clust_info

    if save:
        save_json(disc, save)

def feature_discrepancy(df, cluster_info, f, num_bins = None, weight_bins = False):
    """Finds discrepancy of a feature between all data and a specific cluster by summing the 
    percentages of difference across all values in the feature
    
    df = dataframe
    
    cluster_info = tuple of cluster column name and cluster value
    
    translate = column name to represent the cluster
    
    num_bins = force the data to a certain number of bins (recommended for continuous data)
    
    weight_bins = weight the discrepancy of each value proportionally to the number of possible values 
           so the discrepancy in each value will be worth less if the number of possible values is larger (e.g.,
           discrepancy in a cluster with 5 possible values worth more cluster with 100 possible values)"""
    #find optimal binning later
    
    feat = pd.DataFrame(df[f])[f]
    cluster = pd.DataFrame(df[df[cluster_info[0]] == cluster_info[1]][f])[f]
    
    feature_map = {}
    if num_bins != None and len(set(df[f])) > num_bins:                       
        clustering = k_cluster([[x] for x in feat], k=num_bins)
        for item in range(len(feat)):
            feature_map[feat[item]] = np.float64(clustering["centers"][clustering["clusters"][item]])
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
    cluster_count = dict(cluster_dist)
    
    for item in feat_dist:
        feat_dist[item] = float(feat_dist[item])/len(feat)

    for item in cluster_dist:
        cluster_dist[item] = float(cluster_dist[item])/len(cluster)
    
    diff_arr = [(item, abs(feat_dist[item] - cluster_dist[item]), feat_dist[item], cluster_dist[item]) for item in feat_dist]

    top_arr = [(item[0], item[1], cluster_count[item[0]], item[2], item[3]) for item in sorted(diff_arr, key = lambda x: x[1], reverse = True)][:5]
    
    weighted_bins = float(1)
    if weight_bins:
        weighted_bins = float(len(diff_arr)) 
        
    for item in diff_arr:
        discrepancy += item[1]/weighted_bins
        
    return discrepancy, top_arr

def order_feature_discrepancy(df, cluster_info, feature_info, max_bins = 30, weight_bins = False):
    
    discrepancy_dict = defaultdict(list)
    for clust in cluster_info[1:]:
        for f in feature_info:
            disc_info = feature_discrepancy(df, (cluster_info[0], clust), 
                        f, weight_bins = weight_bins, num_bins = min(len(set(df[f])), max_bins))
            
            discrepancy_dict[clust].append((f, disc_info[0], disc_info[1]))
            
            discrepancy_dict[clust] = sorted(discrepancy_dict[clust], key = lambda x: x[1], reverse = True)
                                          
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
    
    return {"cluster": cluster_avg, "overall": feat_avg, "discrepancy": cluster_avg - feat_avg}

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
    
    return {"cluster": cluster_avg, "overall": feat_avg, "discrepancy": cluster_avg - feat_avg}


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




