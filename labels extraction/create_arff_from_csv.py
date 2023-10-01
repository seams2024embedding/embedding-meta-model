import pandas as pd
import arff
import mat4py as mp
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import os

def create_arff_files(metrics, output_path, oversampling=False, save_new_mat=False):
    metrics.replace({"Bugged": {True: 1, False: -1}}, inplace=True)
    dict_final = {}
    i = 0
    j = 1
    curr_name = ""
    dict_final['res'] = []

    for index, row in metrics[['Project', 'Version']].drop_duplicates().iterrows():
        project = row['Project']
        version = row['Version']
        curr_version = metrics[metrics['Version'] == version]
        curr_version = curr_version.dropna()
        curr_version.drop(columns=['Project', 'Version', 'File','Class'], inplace=True)
        curr_version = curr_version[['Bugged','NumberOfFields','NumberOfPublicFields','NumberOfPublicMethods_Designite','NumberOfChildren','DepthOfInheritance','LOCClass','LCOM','FANIN','FANOUT','TotalNumberOfOperators','NumberOfDistinctOperators','TotalNumberOfOperands','NumberOfDistinctOperands','Length','Vocabulary','Volume','Difficulty','Effort','CBO','WMC_CK','RFC','LOCMethod_CK','Returns','NumberOfVariables','NumberOfParameters_CK','NumberOfLoops','NumberOfComparisons','NumberOfTryCatch','NumberOfParenthesizedExps','NumberOfStringLiterals','NumberOfNumbers','NumberOfAssignments','NumberOfMathOperations','MaxNumberOfNestedBlocks','NumberOfAnonymousClasses','NumberOfInnerClasses','NumberOfLambdas','NumberOfUniqueWords','NumberOfModifiers','NumberOfLogStatements']]
        curr_version = curr_version[curr_version.columns.drop('Bugged').to_list() + ['Bugged']]

        if oversampling:
            try:
                sm = SMOTE(random_state=42,k_neighbors=2)
                src_y = curr_version['Bugged']
                src_x = curr_version.drop(columns=['Bugged'])
                X_samp, y_samp = sm.fit_resample(src_x, src_y)
                X_samp['Bugged'] = y_samp
                curr_version = X_samp
            except:
                print(project, version, "problem with n_neighbors")

        if "/" in str(version):
            version = version.replace("/", "-")
        scaler = MinMaxScaler()
        ls_cols = curr_version.columns.to_list()
        ls_cols.remove('Bugged')
        df = pd.DataFrame(scaler.fit_transform(curr_version.drop(columns=['Bugged'])),
                          columns=ls_cols)
        df['Bugged'] = curr_version['Bugged'].reset_index()['Bugged']
        df = df.round(3)

        arff.dump(f"{output_path}\{project}_{version}.arff", row_iterator=df.values, names=df.columns,
                  relation=project + "_" + str(version))

        if curr_name != project:
            i += 1
            j = 1
        curr_name = project
        dict_final['res'].append((i, j))
        j += 1
    if save_new_mat:
        mp.savemat('projects.mat', dict_final)

        
        
metrics = pd.read_csv("data\\metrics_datasets.csv")
create_arff_files(metrics, "data\\arff_source", oversampling=True, save_new_mat=True)
create_arff_files(metrics, "data\\arff_dest", oversampling=False, save_new_mat=False)
