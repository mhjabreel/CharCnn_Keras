class DataConfig(object):
    """
    Parameters for dataset:

    """
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    alphabet_size = len(alphabet)
    input_size = 1014
    num_of_classes = 4
    training_data_source = 'data/ag_news_csv/train.csv'
    validation_data_source = 'data/ag_news_csv/test.csv'

    def __init__(self):
        print("Data Configurations loaded")


class TrainingConfig(object):
    """
    Parameters for training pipeline:

    """
    base_rate = 1e-2
    momentum = 0.9
    decay_step = 15000
    decay_rate = 0.95
    epochs = 5000
    batch_size = 128
    evaluate_every = 100
    checkpoint_every = 100

    def __init__(self):
        print("Training Configurations loaded")


class ZhangCharCNNConfig(object):
    """
    Parameters for Character Level CNN model described in Zhang et al., 2015:
        embedding_size (int): Size of embeddings
        conv_layers (list[list[int]]): List of Convolution layers for model
                                       in format [num_filters, filter_width, pool_size]
        fully_connected_layers (list[int]): List of Fully Connected layers for model
        threshold (float): Threshold value for ThresholdedReLU activation function
        dropout_p (float): Dropout Probability
        optimizer (str): Training optimizer
        loss (str): Loss function

    """
    embedding_size = 128
    conv_layers = [[256, 7, 3],
                   [256, 7, 3],
                   [256, 3, None],
                   [256, 3, None],
                   [256, 3, None],
                   [256, 3, 3]]
    fully_connected_layers = [1024, 1024]
    threshold = 1e-6
    dropout_p = 0.5
    optimizer = 'adam'
    loss = 'categorical_crossentropy'

    def __init__(self):
        print("ZhangCharCNN Configurations loaded")


class KimCharCNNConfig(object):
    """
    Parameters for Character Level CNN model described in Kim et al., 2015:
        embedding_size (int): Size of embeddings
        conv_layers (list[list[int]]): List of Convolution layers for model
                                       in format [num_filters, filter_width]
        fully_connected_layers (list[int]): List of Fully Connected layers for model
        dropout_p (float): Dropout Probability
        optimizer (str): Training optimizer
        loss (str): Loss function

    """
    embedding_size = 128
    conv_layers = [[256, 10],
                   [256, 7],
                   [256, 5],
                   [256, 3]]
    fully_connected_layers = [1024, 1024]
    dropout_p = 0.1
    optimizer = 'adam'
    loss = 'categorical_crossentropy'

    def __init__(self):
        print("KimCharCNN Configurations loaded")
