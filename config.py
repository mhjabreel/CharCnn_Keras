class DataConfig(object):
    """
    Parameters for dataset.
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    alphabet_size = len(alphabet)
    input_size = 1014
    num_of_classes = 4
    training_data_source = 'data/ag_news_csv/train.csv'
    validation_data_source = 'data/ag_news_csv/test.csv'


class CharCNNConfig(object):
    """
    Parameters for Character Level CNN model.
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


class TrainingConfig(object):
    """
    Parameters for training pipeline.
    """
    base_rate = 1e-2
    momentum = 0.9
    decay_step = 15000
    decay_rate = 0.95
    epochs = 5000
    batch_size = 128
    evaluate_every = 100
    checkpoint_every = 100
