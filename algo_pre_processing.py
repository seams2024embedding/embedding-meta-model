import pandas as pd
from scipy.io import loadmat
import os
import numpy as np
class ProjectsData:
    """
    Class for processing projects data.

    Attributes:
        MCW_path (str): Path to MCW algorithm results.
        projects_path (str): Path to project files.
        features_path (str): Path to features dataframe.
        algo_matlab_results (str): Path to other algorithm results in MATLAB format.
        to_use (list): List of indices to use for processing data.
        list_of_dfs (list): List of dataframes.
        algo_list (list): List of algorithm names.
        algo_scores (list): List of algorithm scores.

    Methods:
        get_data(label_method='max', basic_features=None): Process the data and return the joined dataframe.
        get_mode_value(final_results): Get the mode value for each project and create new labels.
        get_max_fs_value(final_results): Get the best and second best algorithms for each version.
        get_max_value(final_results): Get the best algorithm for each version.
        get_algo_list(): Get the list of algorithms.
        get_algo_scores(): Get the list of algorithm scores.
    """

    def __init__(self,MCW_path= 'MCW', projects_path= 'Projects_Arff', features_path= 'embeddings/projects_embedding_base_new.csv',
                 algo_matlab_results= 'Results_Rest', to_use=None):
        """
        Initialize the ProjectsData object.

        Args:
            MCW_path (str, optional): Path to MCW algorithm results. Defaults to 'MCW'.
            projects_path (str, optional): Path to project files. Defaults to 'Projects_Arff'.
            features_path (str, optional): Path to features dataframe. Defaults to 'embeddings/projects_embedding_base_new.csv'.
            algo_matlab_results (str, optional): Path to other algorithm results in MATLAB format. Defaults to 'Results_Rest'.
            to_use (list, optional): List of indices to use for processing data. Defaults to None.
        """
        self.MCW_path = MCW_path
        self.projects_path = projects_path
        self.features_path = features_path
        self.algo_matlab_results = algo_matlab_results
        # list of runs to use
        self.list_of_dfs = []
        if to_use:
            self.to_use = to_use
        else:
            self.to_use = list(range(1, 41)) + list(range(42, 51))
        self.algo_list = []
        self.algo_scores = []

    def get_data(self, label_method='max', basic_features=None):
        """
        Process the data and return the joined dataframe.

        Args:
            label_method (str, optional): Label extraction method. Defaults to 'max'.
            basic_features (bool, optional): Flag to indicate if only basic features should be included. Defaults to None.

        Returns:
            pd.DataFrame: Joined dataframe of features and labels.
        """

        MCW_path = self.MCW_path
        projects_path = self.projects_path
        features_path = self.features_path
        algo_matlab_results = self.algo_matlab_results
        # getting the results of MCW algorithm
        res1 = pd.DataFrame()
        for num in self.to_use:
            results_df = pd.read_csv(f'{MCW_path}/MCW_res{num}.csv')
            results_df['pv'] = results_df['project'] + "_" + results_df['version']
            results_df = results_df[['pv', 'f1_score','precision','recall','accuracy']]
            results_df.columns = ['pv', 'MCW','MCW_precision','MCW_recall','MCW_accuracy']
            res1 = pd.concat([res1, results_df])
        self.algo_scores = ['MCW','MCW_precision','MCW_recall','MCW_accuracy']
        self.list_of_dfs.append(res1)

        # getting the results of the other algorithms
        files = {'Dycom': 'results_mean', 'TDS': 'results', 'LT': 'results', 'TCA_rnd': 'results_mean', 'TPTL': 'tptl'}
        self.algo_list = ['MCW'] + list(files.keys())
        col_names = []
        for file, res_name in files.items():
            res = pd.DataFrame()
            for i in self.to_use:
                mat = loadmat(f'{algo_matlab_results}/{file}/{file}{i}.mat')
                file_names = [file[:-5] for file in os.listdir(f'{projects_path}/{i}')]
                # f1,precision,recall,accuracy
                df = pd.DataFrame(mat[res_name], columns=[file, file+"_precision", file+"_recall", file+"_accuracy"])
                df['pv'] = file_names

                res = pd.concat([res, df[['pv', file, file+"_precision", file+"_recall", file+"_accuracy"]]])
            self.algo_scores += [file, file + "_precision", file + "_recall", file + "_accuracy"]
            self.list_of_dfs.append(res[['pv',file, file + "_precision", file + "_recall", file + "_accuracy"]])

        print("Loading projects is finished...")
        for i in range(0,6):
            self.list_of_dfs[i]['pv'] = self.list_of_dfs[i]['pv'].str.replace("/", "-")

        new_list = [self.list_of_dfs[0].reset_index().drop(columns=['index'])]
        new_list = new_list+ [self.list_of_dfs[i].loc[self.list_of_dfs[i]['pv'].isin(self.list_of_dfs[0]['pv'].to_list())].reset_index().drop(columns=['index','pv']) for i in range(1,6)]
        # merge all the results into one dataframe
        final_results = pd.concat(new_list, axis=1)
        final_results.columns = ['pv'] +self.algo_scores
        
        ## LABEL EXTRACTION
        if label_method == 'max':
            final_results = self.get_max_value(final_results)
            
        if label_method == 'max_first_second':
            final_results = self.get_max_fs_value(final_results)

        if label_method == 'mode':
            final_results = self.get_mode_value(final_results)
                    
        # get the features dataframe and join it with the labels
        features = pd.read_csv(f"{features_path}")
        features['pv'] = features['pv'].str.replace("/", "-")
        joined = features.merge(final_results, on='pv', how='inner')
        joined = joined.reset_index().drop(columns=['index'])
        if basic_features:
            # just first and last version - for BASIC FEATURES
            joined[['project', 'version']] = joined.pv.str.split("_", expand=True)[[0, 1]]
            project = ''
            tmp_project = pd.DataFrame()
            res = pd.DataFrame()
            for index, row in joined.iterrows():
                if row['project'] != project:
                    res = pd.concat([res, pd.DataFrame(row).T])
                    if project != '':
                        res = pd.concat([res, pd.DataFrame(tmp_project).T])
                    project = row['project']
                tmp_project = row
            joined = res
            
        return joined
    
    
    def get_mode_value(self, final_results):
        """
        Get the mode value for each project and create new labels.

        Args:
            final_results (pd.DataFrame): DataFrame with algorithm results.

        Returns:
            pd.DataFrame: DataFrame with mode values and new labels.
        """
        # get the best algorithm for each version
        final_results[['project', 'version']] = final_results.pv.str.split("_", expand=True)[[0, 1]]
        final_results['best_algo'] = final_results[self.algo_list].idxmax(axis=1)
        # create dataframe of mode values for each project
        mode_values = pd.DataFrame(final_results.groupby('project')['best_algo'].agg(pd.Series.mode))
        mode_label2, mode_label = [], []
        for index, row in mode_values.iterrows():
            mode_val = mode_values.loc[index].values[0]
            # if there is more than one mode value, get the second value (randomly)
            if type(mode_val) != str:
                mode_label2.append(mode_val[1])
                mode_label.append(mode_val[0])
            else:
                mode_label2.append(mode_val)
                mode_label.append(mode_val)
        # create new labels - based on the mode value for each version
        mode_values['mode_algo'] = mode_label2
        mode_values = mode_values.drop(columns=['best_algo'])
        final_results = final_results.drop(columns=['version'])
        final_results = mode_values.reset_index().merge(final_results, on=['project'])
        final_results = final_results.drop(columns=['project'])
        final_results = final_results[final_results['mode_algo'] == final_results['best_algo']]

        return final_results
    
    
    def get_max_fs_value(self,final_results):
        """
        Get the best and second best algorithms for each version.

        Args:
            final_results (pd.DataFrame): DataFrame with algorithm results.

        Returns:
            pd.DataFrame: DataFrame with best algorithm labels.
        """
        df = final_results[['pv','MCW', 'Dycom', 'TDS', 'LT', 'TCA_rnd', 'TPTL']]
        df = df.set_index('pv')
        # get best and second best algorithms
        first = pd.DataFrame(df.apply(lambda row: row.nlargest(1).values[-1], axis=1), columns=['first_score'])
        second = pd.DataFrame(df.apply(lambda row: row.nlargest(2).values[-1], axis=1), columns=['second_score'])
        # merge results
        df = df.merge(first, left_index=True, right_index=True)
        df = df.merge(second, left_index=True, right_index=True)
        df['first_bes'] = df[['MCW', 'Dycom', 'TDS', 'LT', 'TCA_rnd', 'TPTL']].T.apply(lambda x: x.nlargest(1).idxmin())
        df['second_bes'] = df[['MCW', 'Dycom', 'TDS', 'LT', 'TCA_rnd', 'TPTL']].T.apply(lambda x: x.nlargest(2).idxmin())
        df['diff'] = df['first_score'] - df['second_score']
        first_vals = df.copy()
        first_vals['best_algo'] = first_vals['first_bes']
        first_vals = first_vals.reset_index()[['pv','best_algo']]
        mask = df['diff'].lt(0.01)
        second_vals = df[mask].copy()
        second_vals['best_algo'] = second_vals['second_bes']
        second_vals = second_vals.reset_index()[['pv','best_algo']]
        both = pd.concat([first_vals,second_vals])
        final_results = both.merge(final_results, on='pv', how='inner')
        return final_results
    
    def get_max_value(self,final_results):
        """
        Get the best algorithm for each version.

        Args:
            final_results (pd.DataFrame): DataFrame with algorithm results.

        Returns:
            pd.DataFrame: DataFrame with best algorithm labels.
        """
        final_results['best_algo'] = final_results[self.algo_list].idxmax(axis=1)
        final_results = final_results.reset_index().drop(columns=['index'])
        return final_results

    
    def get_algo_list(self):
        """
        Get the list of algorithms.

        Returns:
            list: List of algorithm names.
        """
        return self.algo_list

    
    def get_algo_scores(self):
        """
        Get the list of algorithm scores.

        Returns:
            list: List of algorithm scores.
        """
        return  self.algo_scores