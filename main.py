import tensorflow as tf

from models.char_cnn_zhang import CharCNNZhang
from models.char_cnn_kim import CharCNNKim

from data_utils import Data

from config import DataConfig
from config import ZhangCharCNNConfig
from config import KimCharCNNConfig
from config import TrainingConfig

tf.flags.DEFINE_string("model", "zhang", "Specifies which model to use. (default: 'zhang')")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

if __name__ == "__main__":
    # Load data configurations
    data_config = DataConfig()
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

    # Load model configurations and build model
    if FLAGS.model == "zhang":
        model_config = ZhangCharCNNConfig()
        model = CharCNNZhang(input_size=data_config.input_size,
                             alphabet_size=data_config.alphabet_size,
                             embedding_size=model_config.embedding_size,
                             conv_layers=model_config.conv_layers,
                             fully_connected_layers=model_config.fully_connected_layers,
                             num_of_classes=data_config.num_of_classes,
                             threshold=model_config.threshold,
                             dropout_p=model_config.dropout_p,
                             optimizer=model_config.optimizer,
                             loss=model_config.loss)
    elif FLAGS.model == "kim":
        model_config = KimCharCNNConfig()
        model = CharCNNKim(input_size=data_config.input_size,
                           alphabet_size=data_config.alphabet_size,
                           embedding_size=model_config.embedding_size,
                           conv_layers=model_config.conv_layers,
                           fully_connected_layers=model_config.fully_connected_layers,
                           num_of_classes=data_config.num_of_classes,
                           dropout_p=model_config.dropout_p,
                           optimizer=model_config.optimizer,
                           loss=model_config.loss)

    # Load training configurations
    training_config = TrainingConfig()
    # Train model
    model.train(training_inputs=training_inputs,
                training_labels=training_labels,
                validation_inputs=validation_inputs,
                validation_labels=validation_labels,
                epochs=training_config.epochs,
                batch_size=training_config.batch_size,
                checkpoint_every=training_config.checkpoint_every)
