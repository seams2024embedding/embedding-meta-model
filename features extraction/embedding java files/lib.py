def create_embedding_from_file(filename, model, version_path):
    """
    Creates an embedding from a file containing Java code.

    Args:
        filename (str): The path to the file containing Java code.
        model: The language model used for generating embeddings.
        version_path (str): The path to the version directory where the embeddings will be saved.

    Returns:
        model: The input model.

    Raises:
        UnicodeDecodeError: If there is an error decoding the file as Unicode.

    """

    file = filename.split('/')[-1].split('.')[0]
    with open(filename) as f:
        try:
            contents = f.read()
        except UnicodeDecodeError:
            print(f'unicode decode error, skiping {filename} file') 
        try:
            java_code = contents.split('*/')[-1]
        except:
            java_code = contents
        input_ids = tokenizer(java_code, return_tensors="pt").input_ids.to(device)
        output = model(input_ids).last_hidden_state
        pooling = MaxPool2d((output.shape[1],1))
        pool = pooling(output)
        pool = pool.squeeze(0)
        pickle.dump(pool, open(f'{version_path}/{OUTPUT_PATH}/{file}.pickle', 'wb'))
    return model