from imblearn.over_sampling import SMOTE
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from adapt.instance_based import KMM
from sklearn.linear_model import LogisticRegressionCV
from cvxopt import matrix
from cvxopt.modeling import variable, op, max, sum


class MCW_Algorithm:

    def __init__(self):
        self.src = None
        self.dst = None
        self.clustered = None
        self.models = None
        self.m_weights = None
        self.sampled_target = None

    def select_src_target(self, df, keys, target_col, random_state=None,ind=0):
        '''
        Select source and target projects for the algorithm
        :param df: the df that contains the projects' data
        :param keys: the unique key that represent project
        :param target_col: the name of the target column
        :param random_state:
        :return: src_x, src_y, dst_x, dst_y
        '''
        # randomly select source and target projects
        z = df[keys].drop_duplicates().reset_index()
        source_projects = z.drop(ind)
        selected_project = source_projects.sample(1, random_state=random_state)
        src_project = dict(selected_project.iloc[0, :][['Project','Version']])
        dst_project = dict(z.iloc[ind, :][['Project','Version']])

        # get the source data
        src_project_data = df.loc[(df[list(keys)] == pd.Series(src_project)).all(axis=1)]
        src_project_data.drop(columns=keys, inplace=True)
        src_project_data = src_project_data[src_project_data.columns.drop(target_col).to_list() + [target_col]]
        self.src = src_project_data

        # get the target data
        dst_project_data = df.loc[(df[list(keys)] == pd.Series(dst_project)).all(axis=1)]
        dst_project_data.drop(columns=keys, inplace=True)
        dst_project_data = dst_project_data[dst_project_data.columns.drop(target_col).to_list() + [target_col]]
        self.dst = dst_project_data

        # split x y
        src_y = src_project_data[target_col]
        src_x = src_project_data.drop(columns=[target_col])
        dst_y = dst_project_data[target_col]
        dst_x = dst_project_data.drop(columns=[target_col])

        return src_x, src_y, dst_x, dst_y

    def sampled_target_data(self,target_data,target_col):
        # Extract 5% of the target data
        sampled_target = target_data.sample(frac=0.05, random_state=5)
        if len(sampled_target) < 20:
            sampled_target = target_data.sample(frac=0.2, random_state=5)
        sampled_target['weight'] = 1
        src_y = sampled_target[target_col]
        src_x = sampled_target.drop(columns=[target_col])
        try:
            sm = SMOTE(random_state=42,k_neighbors=1)
            X_samp, y_samp = sm.fit_resample(src_x, src_y)
            X_samp[target_col] = y_samp
        except:
            X_samp = sampled_target
        return X_samp

    def fit(self, src_x, src_y, target_data, target_col,random_state):
        '''
        Fit the MCW algorithm
        :param random_state: random_state variable
        :param src_x: the source project x data
        :param src_y: the source project y data
        :param target_data: the target data df (x+y)
        :param target_col: the target column name
        :return:
        '''
        # STEP 1: Divide Ps into multiple components
        clustered_df = self.clustering(src_x, n_clusters=4, random_state=random_state, method="spectral", n_components=4,
                                       assign_labels='cluster_qr')
        clustered_df[target_col] = src_y

        # STEP 2.1: Reweighing the instances of each component by KMM
        # Split the source data into components (based on the clusters), apply KMM
        list_of_components = self.KMM(clustered_df, target_data)

        sampled_target = self.sampled_target_data(target_data, target_col)
        # Add the target data to each component
        list_of_components = self.add_target_data(list_of_components, sampled_target)
        too_small = [df for df in list_of_components if len(df)<20]
        if too_small != []:
            temp = []
            for df in list_of_components:
                if len(df)>=20:
                    temp.append(df)
            list_of_components = temp
            list_of_components.append(pd.concat(too_small))

        # STEP 3.1: Train based classifier for each component
        models, predictions, scores = self.train_base_classifiers(list_of_components, target_col)

        # STEP 3.2: Assembling the classifiers and optimise their weights with Pl
        # initial weights
        self.assign_weights(scores, target_col, predictions)

    def clustering(self, src_project_data=None, n_clusters=8, random_state=None, method="spectral", n_components=None,
                   assign_labels='kmeans'):
        '''
        Cluster the source data into clusters (components)
        :param src_project_data: the data of the source project
        :param n_clusters: select the number of clusters to divide the data into
        :param random_state:
        :param method: Clustering method: Spectral clustering - "spectral", Kmeans - "kmeans", Kmeans minibatch - "kmeans_minibatch"
        :param n_components: number of components
        :param assign_labels: The strategy for assigning labels in the embedding space
        :return: the clustered data (pandas df)
        '''
        if src_project_data is None:
            src_project_data = self.src

        if method == "spectral":
            sc = SpectralClustering(n_clusters=n_clusters, random_state=random_state, n_components=n_components,
                                    assign_labels=assign_labels,affinity='nearest_neighbors',n_neighbors=2)
        elif method == "kmeans":
            sc = KMeans(n_clusters=n_clusters, random_state=random_state)
        elif method == "kmeans_minibatch":
            sc = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state)
        x = sc.fit(src_project_data)
        src_project_data['cluster'] = pd.Series(x.labels_, index=src_project_data.index)
        self.clustered = src_project_data
        return src_project_data

    def KMM(self, clustered_df, target_data):
        '''
        Apply KMM on each cluster
        :param clustered_df: the clustered dataframe
        :param target_data: the target data
        :return: the components with the weights for each instance in each cluster
        '''
        # list of the clusters
        clusters = clustered_df['cluster'].drop_duplicates().to_list()
        list_of_components = []
        for val in clusters:
            # extract the relevant cluster
            curr_cluster = clustered_df[clustered_df['cluster'] == val]

            # reweight the instances of each component by KMM
            kmm = KMM(random_state=1)
            weights = kmm.fit_weights(curr_cluster.drop(columns=['cluster']), target_data)

            # add the weights for each instance in the cluster
            curr_cluster['weight'] = np.array(weights)
            list_of_components.append(curr_cluster.drop(columns=['cluster']))
        return list_of_components

    def add_target_data(self, list_of_components, sampled_target):
        '''
        Add the target data to each component
        :param list_of_components: the clusters
        :param sampled_target: the target data that will be added to each cluster
        :return: the list of components in which each component include the target data
        '''

        with_target = []
        for component in list_of_components:
            with_target.append(component.append(sampled_target))
        self.sampled_target = sampled_target
        return with_target

    def train_base_classifiers(self, list_of_components, target_col):
        '''
        Create a logistic regression model for each component
        :param list_of_components: the components
        :param target_col: the target column name
        :return: list of models (one for each component)
        '''
        scores = []
        predictions = []
        models = []
        for component in list_of_components:
            # split the component into X and y
            X = component.drop(columns=[target_col, "weight"])
            y = component[target_col]
            curr_weights = component["weight"]
            # build the classifier and train it using the weights for each sample
            clf = LogisticRegressionCV(cv=4, random_state=0)
            clf.fit(X, y, sample_weight=curr_weights)
            # get the score of the classifier
            preds = clf.predict(X)
            curr_score = clf.score(X, y, curr_weights)
            scores.append(curr_score)
            models.append(clf)
            # get the predictions of the labeled data (5% from target)
            predictions.append(preds[-len(self.sampled_target):])
        self.models = models
        return models, predictions, scores

    def assign_weights(self, scores, target_col, predictions):
        '''
        Assign the weights of the classifiers
        :param scores: the scores of each classifiers (as calculated in train_base_classifiers method)
        :param target_col: the name of the target column
        :param predictions: the list of predictions for 5% target data
        :return: the weights for each classifier
        '''
        # initial weights for each model
        weights_classifiers = scores / sum(scores)

        # get the prediction for the 5% target data
        yi = []
        for j in range(len(self.sampled_target[target_col])):
            yi.append([row[j] * 1.0 for row in predictions])
        mat_yi = matrix(yi)

        # get the real labels for the 5% target data
        yl = self.sampled_target[target_col]

        # build w, w*, y_i, w_l
        mat_yl = matrix(yl * 1.0)
        w_star = matrix(weights_classifiers)
        yi_trans = mat_yi.trans()
        w = variable(len(w_star))

        # solve the minimization problem
        prob = op(sum(abs(yi_trans * w - mat_yl)) + sum(abs(w - w_star)), [sum(w) == 1, w >= 0])
        prob.solve()

        new_weights = list(prob.variables()[0].value)
        self.m_weights = new_weights

    def predict(self, X_test):
        '''
        Predict the lables of X_test
        :param X_test:
        :return: the prediction for each sample
        '''
        x_test_pred = []
        models = self.models
        models_weights = self.m_weights
        # get the prediction for X_test from each classifier
        for i in range(len(models)):
            predictions_test = models[i].predict(X_test)
            x_test_pred.append(predictions_test)
        # get the predictions for each sample in different list
        preds_test = []
        for j in range(len(x_test_pred[0])):
            preds_test.append([row[j] for row in x_test_pred])

        # sum the predictions for each sample
        combined_preds = []
        for output in preds_test:
            final_res = sum(np.array(output) * np.array(models_weights))
            combined_preds.append(final_res)

        # get the combined prediction
        final_pred = []
        for res in combined_preds:
            final_pred.append(1) if res > 0 else final_pred.append(-1)

        return final_pred

    def print_clusters_2d(self, clustered_df):
        pca = PCA(n_components=2)
        projected = pca.fit_transform(clustered_df)

        # extract x,y vectors
        x = projected[:, 0]
        y = projected[:, 1]
        colors = cm.nipy_spectral(clustered_df['cluster'] / 8)
        plt.figure(figsize=(10, 10))
        plt.scatter(x, y, s=50, lw=0, alpha=0.7, c=colors)
        plt.title("Clustering results")
        plt.legend()
        plt.show()
