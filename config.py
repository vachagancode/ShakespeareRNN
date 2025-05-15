def get_config():
    return {
        "input_size" : 100,
        "hidden_size": 1024,
        "vocab_size": 39, # len(chars)
        "num_layers": 2,
        "dropout": 0.2,

        "annotations_file" : "annotations_file.csv",
        "max_length" : 100,

        "num_epochs" : 100
    }