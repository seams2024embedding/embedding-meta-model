import pandas as pd
import numpy as np

metrics = pd.read_csv('data\\metrics_datasets.csv').sort_values(
    by=['Project', 'Version'])

metrics.replace({"Bugged": {True: 1, False: -1}}, inplace=True)

metrics['pv'] = metrics['Project'] + "_" + metrics['Version']
metrics['pv'] = metrics['pv'].replace("/", "_")
# metrics = metrics.drop(columns=["Project", "File", "Class", "Version", "NPathComplexity", "Bugged"]) #first batch
metrics = metrics.drop(columns=["Project", "File", "Version", "Bugged"]) # second batch
metrics = metrics.dropna()
mean_vals = metrics.groupby(by=['pv']).mean()
mean_vals.columns = [name + "_avg" for name in mean_vals.columns]

std_vals = metrics.groupby(by=['pv']).std()
std_vals.columns = [name + "_std" for name in std_vals.columns]

skew_vals = metrics.groupby(by=['pv']).skew(axis=0)
skew_vals.columns = [name + "_skew" for name in skew_vals.columns]

min_vals = metrics.groupby(by=['pv']).max()
min_vals.columns = [name + "_min" for name in min_vals.columns]
max_vals = metrics.groupby(by=['pv']).max()
max_vals.columns = [name + "_max" for name in max_vals.columns]

df = metrics.groupby(by=['pv']).count()[['NumberOfFields']]
df.columns = ["NumberOfFiles"]

dict_pv = {'pv': [], 'mean_corr': [], 'std_corr': []}

for p_v in metrics['pv'].drop_duplicates():
    dict_pv['pv'].append(p_v)
    c2 = metrics[metrics['pv'] == p_v].corr().copy()
    c2.values[np.tril_indices_from(c2)] = np.nan
    mean = c2.unstack().mean()
    std = c2.unstack().std()
    dict_pv['mean_corr'].append(mean)
    dict_pv['std_corr'].append(std)

corr_data = pd.DataFrame(dict_pv)
corr_data = corr_data.set_index('pv')
std_mean = std_vals.join(mean_vals)
min_max = min_vals.join(corr_data)
min_max_n = min_max.join(skew_vals)
all_features = std_mean.join(min_max_n)
all_features = all_features.join(df)

all_features.to_csv('meta_model_features\\statistical_features\\statistical_meta_features.csv')
