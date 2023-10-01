# Create Embedding Features from the Java Raw data

## Using the Embedding Classes Notebook

To utilize the Embedding Classes Notebook, follow these steps:

### Configuration:

Configure the `config` file by defining:
1. `JAVA_FILES_PATH`: the path of the Java projects. The expected folder structure for the projects is as follows:
- Master folder
    - Project folder
        - Version folder
            - Java files (can be in subfolders, as the notebook uses `glob.iglob` with `recursive=True` to extract all files in the root folder).
2. `MODEL_NAME`: Specify the desired model from the available `Codet5` models in the config file.
3. `OUTPUT_PATH`: Define the output folder path in the config file. The generated embeddings will be saved under the `master/project/version/OUTPUT_PATH` directory.

### Running the Notebook:

Execute the notebook to generate the embeddings.



## Using Create Embedding Features
using this notebook you can create an embedding matrix that will be used as the meta-features for the meta-model. 
1. Configure the `config` file by defining:
    - `OUTPUT_EMBEDDING_FEATURES`: the output path for the meta-features embedding based. 
    - `projects_to_exclude`: if you want to exclude some projects from the final features set. 
2. Run the notebook