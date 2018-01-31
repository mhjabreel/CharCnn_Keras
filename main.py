from models.char_cnn import CharCNN
from data_utils import Data
from config import DataConfig
from config import CharCNNConfig
from config import TrainingConfig


if __name__ == "__main__":
    data_config = DataConfig()
    model_config = CharCNNConfig()
    training_config = TrainingConfig()

    # Load training data
    training_data = Data(data_source=data_config.training_data_source,
                         alphabet=data_config.alphabet,
                         input_size=data_config.input_size,
                         batch_size=0,
                         no_of_classes=data_config.num_of_classes)
    training_data.load_data()
    training_inputs, training_labels = training_data.get_all_data()

    # Load validation data
    validation_data = Data(data_source=data_config.validation_data_source,
                           alphabet=data_config.alphabet,
                           input_size=data_config.input_size,
                           batch_size=0,
                           no_of_classes=data_config.num_of_classes)
    validation_data.load_data()
    validation_inputs, validation_labels = validation_data.get_all_data()

    # Load model
    model = CharCNN(input_size=data_config.input_size,
                    alphabet_size=data_config.alphabet_size,
                    embedding_size=model_config.embedding_size,
                    conv_layers=model_config.conv_layers,
                    fully_connected_layers=model_config.fully_connected_layers,
                    num_of_classes=data_config.num_of_classes,
                    threshold=model_config.threshold,
                    dropout_p=model_config.dropout_p,
                    optimizer=model_config.optimizer,
                    loss=model_config.loss)

    # Train model
    model.train(training_inputs=training_inputs, 
                training_labels=training_labels, 
                validation_inputs=validation_inputs,
                validation_labels=validation_labels,
                epochs=training_config.epochs,
                batch_size=training_config.batch_size)
