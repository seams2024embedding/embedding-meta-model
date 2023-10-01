import os
import shutil
import pandas as pd
import random

FEATURES_LIST = ['Project', 'Version','Bugged', 'NumberOfFields', 'NumberOfPublicFields', 'NumberOfPublicMethods_Designite', 'NumberOfChildren',
         'DepthOfInheritance', 'LOCClass', 'LCOM', 'FANIN', 'FANOUT', 'TotalNumberOfOperators',
         'NumberOfDistinctOperators', 'TotalNumberOfOperands', 'NumberOfDistinctOperands', 'Length', 'Vocabulary',
         'Volume', 'Difficulty', 'Effort', 'CBO', 'WMC_CK', 'RFC', 'LOCMethod_CK', 'Returns', 'NumberOfVariables',
         'NumberOfParameters_CK', 'NumberOfLoops', 'NumberOfComparisons', 'NumberOfTryCatch',
         'NumberOfParenthesizedExps', 'NumberOfStringLiterals', 'NumberOfNumbers', 'NumberOfAssignments',
         'NumberOfMathOperations', 'MaxNumberOfNestedBlocks', 'NumberOfAnonymousClasses', 'NumberOfInnerClasses',
         'NumberOfLambdas', 'NumberOfUniqueWords', 'NumberOfModifiers', 'NumberOfLogStatements']
METRICS_PATH = 'data\\metrics_datasets.csv'
NUMBER_OF_GROUPS = 50
def random_groups_arff():
    metrics = pd.read_csv(METRICS_PATH).sort_values(by=['Project', 'Version'])
    metrics = metrics[FEATURES_LIST]
    groups = {1: [], 2: [], 3: [], 4: [], 5: []}
    curr_project = None
    for index, row in metrics[['Project', 'Version']].drop_duplicates().iterrows():
        if curr_project != row['Project']:
            i = 1
            curr_project = row['Project']
        version = row['Version']
        if "/" in version:
            version = version.replace("/", "-")

        groups[i].append(f"{curr_project}_{version}.arff")
        i += 1

    def partition(list_in, n):
        random.shuffle(list_in)
        return [list_in[i::n] for i in range(n)]

    all_groups = []
    for i in range(1, 6):
        all_groups.extend(partition(groups[i], 10))

    source_path = 'data\\arff_source'
    dest_path = 'data\\arff_dest'
    for i in range(1, NUMBER_OF_GROUPS+1):
        os.mkdir(f'{dest_path}\\{i}')
        os.mkdir(f'{source_path}\\{i}')
        for file in all_groups[i]:
            shutil.copy(f'{dest_path}\\{file}', f'{dest_path}\\{i}\\{file}')
            shutil.copy(f'{source_path}\\{file}', f'{source_path}\\{i}\\{file}')


def random_groups_csv():
    metrics = pd.read_csv(METRICS_PATH).sort_values(by=['Project', 'Version'])
    source_path = 'data\\arff_source'
    metricsdf = pd.read_csv(METRICS_PATH).sort_values(by=['Project', 'Version'])[
        ['Project', 'Version']].drop_duplicates().reset_index()
    metrics = metrics[FEATURES_LIST]

    for i in range(1, NUMBER_OF_GROUPS+1):
        curr_csv = pd.DataFrame()
        for file in os.listdir(f'{source_path}\\{i}'):
            file_name = file[:-5]
            splitted = file_name.split("_")
            project = splitted[0]
            version = "_".join(splitted[1:])
            only_project = metrics.loc[(metrics['Project'] == project) & (metrics['Version'] == version)]
            if len(only_project) == 0:
                version = version.replace("-", "/")
                dict_replacements = {"rel/commons/csv/1.0": "rel/commons-csv-1.0",
                                     "rel/wicket/6.23.0": "rel/wicket-6.23.0",
                                     "releases/lucene/solr/4.10.4": "releases/lucene-solr/4.10.4",
                                     'interim/isis/1.15.1.20171221/1739': 'interim/isis-1.15.1.20171221-1739',
                                     'rel/commons/vfs/2.0': 'rel/commons-vfs-2.0',
                                     'rel/commons/vfs/2.2': 'rel/commons-vfs-2.2',
                                     'rel/commons/csv/1.2': 'rel/commons-csv-1.2', 'rel/nifi/0.6.1': 'rel/nifi-0.6.1',
                                     'interim/isis/1.16.0.20180130/1145': 'interim/isis-1.16.0.20180130-1145',
                                     'rel/nifi/0.7.0': 'rel/nifi-0.7.0', 'rel/commons/csv/1.5': 'rel/commons-csv-1.5',
                                     'interim/isis/1.16.1.20180316/1549': 'interim/isis-1.16.1.20180316-1549',
                                     'rel/commons/vfs/2.4.1': 'rel/commons-vfs-2.4.1',
                                     'rel/nifi/1.4.0': 'rel/nifi-1.4.0',
                                     'rel/commons/vfs/2.5.0': 'rel/commons-vfs-2.5.0',
                                     'rel/commons/csv/1.6': 'rel/commons-csv-1.6', 'rel/isis/1.12.2': 'rel/isis-1.12.2',
                                     'rel/calcite/1.4.0/incubating':'rel/calcite-1.4.0-incubating',
                                     'rel/isis/1.7.0':'rel/isis-1.7.0','rel/nifi/1.9.2':'rel/nifi-1.9.2',
                                     'rel/commons/csv/1.7':'rel/commons-csv-1.7','rel/calcite/1.5.0':'rel/calcite-1.5.0'
                                     }
                if version in dict_replacements.keys():
                    version = dict_replacements[version]
                only_project = metrics.loc[(metrics['Project'] == project) & (metrics['Version'] == version)]
                index = metricsdf.loc[(metricsdf['Project'] == project) & (metricsdf['Version'] == version)].index[0]
                only_project['index'] = index

                if len(only_project) == 0:
                    print(f"project: {project}, version: {version}")
                    print("still not working")
            else:
                index = metricsdf.loc[(metricsdf['Project'] == project) & (metricsdf['Version'] == version)].index[0]
                only_project['index'] = index
            curr_csv = pd.concat([curr_csv, only_project])
        curr_csv.to_csv(f'algo repository\\Qiu et al.2020\\input\\MCW{i}.csv')
        print(f"csv number {i} saved")

random_groups_arff()
random_groups_csv()