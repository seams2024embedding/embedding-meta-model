# Steps to replicate the creation and evaluation of embedding-based:

## 1. Create Data Folder:
In the `data` directory, set up the following folders:
- `java_files`: This folder should contain a repository of Java projects. The structure should be `java_files/project/version`. The order and hierarchy of the files within this directory are not important. We will iterate over all the files and use only the `.java` files.
- `Project_Arff`: Place all the ARFF format files in this directory. These files are created based on a CSV file of features (instructions on creating them will be provided separately in section 2).
- `Results_Rest`: Save all the results of the different algorithms from the algorithm repository in this directory. Instructions on obtaining the results will be provided separately in section 2. This path uses the MATLAB algorithm repository. Based on: https://bitbucket.org/ChaoLiuCQ/replication-kit-ist2018-tptl
- `MCW`: Save all the results of the MCW algorithm in this directory. Instructions on obtaining the results will be provided separately in section 2.

## 2. Extracting the Labels:
Please refer to the detailed readme file located in the `labels extraction` folder for instructions on extracting the labels, located [here](/labels%20extraction/readme.md).


## 3. Features Extraction:
- **Statistical Meta Features:** Execute the `statistical meta features.py` script to generate the statistical meta features.
- **Embedding-based Meta Features:** Please refer to the detailed readme file located in the `embedding java files` folder for instructions on extracting embedding-based meta features from Java files, located [here](features%20extraction/embedding%20java%20files/readme.md).


## 4. Run Meta Model:
Now that you have completed the necessary steps for the meta model, you can run the `meta-model.ipynb` notebook. The notebook is based on the `algo_pre_processing.py` file.

## General Comment:
Our data is divided into two batches. In the notebook, we use two directories and perform two readings of the features and labels. However, for your own data, you can choose to use only one directory. Instructions specific to your data are provided in detail in the `meta-model.ipynb` notebook, in section 4.

### Data Repo:
**1. Metrics_datasets.csv (both batches):**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8147633.svg)](https://doi.org/10.5281/zenodo.8147633)

**2. Data folder that contains:**
1. MCW (batch 1 & batch 2) results
2. Results_Rest (batch 1 & batch 2) - from MATLAB algorithms.
3. Project Arff (batch 1 & batch 2)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8150398.svg)](https://doi.org/10.5281/zenodo.8150398)

**3. Java source code:**

As mentioned we used 2 batches of data for the analysis, attached is the source code (java source code) of these batches. To get the complete zip file you should download both part and merge the zip files:

Batch 1:
- Part 1: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8150412.svg)](https://doi.org/10.5281/zenodo.8150412)
- Part 2: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8150885.svg)](https://doi.org/10.5281/zenodo.8150885)

Batch 2:
- Part 1: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8152041.svg)](https://doi.org/10.5281/zenodo.8152041)
- Part 2: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8152156.svg)](https://doi.org/10.5281/zenodo.8152156)



