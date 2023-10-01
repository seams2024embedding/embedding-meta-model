# Extracting the labels
To extract the labels for the project versions, follow these steps:

### 1. Get Metrics Dataset
Obtain features about all the relevant Java files for each version of each project using the provided `metrics_datasets.csv` file.

### 2. Run `create_arff_from_csv.py`
   - Run the `create_arff_from_csv.py` Python script to create the `Project_Arff` folder. This folder contains arff files for each version, with the required features in the format required by the MATLAB repository algorithms.
   - The script creates two folders: `arff_source` and `arff_dest`. For each project-version, two files are generated:
     1. The project-version is the source of different project-version bug prediction (with oversampling).
     2. The project-version is the destination project (without oversampling).
   - You can use one of these folders (`arff_source` or `arff_dest`) as the `Project_Arff` folder by renaming it accordingly when running the final notebook of the meta-model.
   - The script also creates `projects.mat` that needs to be located in each benchmark algorithm in the MATLAB algorithm repository.

### 3. Split the data to random groups using `random_groups.py`
   - To ensure efficient memory usage with the MATLAB repository algorithms, a smaller sample of 10 projects is used for each algorithm run.
   - Execute the `random_groups.py` Python script to randomly separate the projects into different groups, while ensuring that **within the same group, one version cannot be the source and another version of the same project cannot be the target.**

### 4. Run each of the MATLAB repository algorithms to get the results of each algorithm for each project-version. 
   - Run all the projects using all the algorithms in the MATLAB repository. The relevant algorithms are located in the `algo repository\Liu et al. 2019` path.
   - For each group (1 to `NUMBER_OF_GROUPS`), run all the algorithms.
   - In every run:
       - Replace the files in the `Promise` folder with the `dest_arff` folder files for the relevant run number (`i`).
       - Replace the `PromiseSource` folder files with the `source_arff` folder files for the relevant run number (`i`).
       - Replace the `index.mat` file with a matrix containing the index `i` of the current run.
   - After running all the projects with all the algorithms, copy the `Results` folder to the following path: `data/Results_Rest`.


### 5. Run the MCW algorithm.
   - The MCW algorithm is located in the `algo repository\Qiu et al.2020` path.
   - Ensure that the `metrics_dataset.csv` file is in the folder.
   - Run the loop over the different projects using the MCW algorithm.
   - After running all the projects with the algorithm, copy the `results_all` folder to the following path: `data/MCW`.
