from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Activation
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from keras.layers import Embedding
from keras.layers import ThresholdedReLU
from keras.layers import Dropout


class CharCNN(object):
    def __init__(self, input_size, alphabet_size, embedding_size,
                 conv_layers, fully_connected_layers, num_of_classes,
                 threshold, dropout_p,
                 optimizer='adam', loss='categorical_crossentropy'):
        self.input_size = input_size
        self.alphabet_size = alphabet_size
        self.embedding_size = embedding_size
        self.conv_layers = conv_layers
        self.fully_connected_layers = fully_connected_layers
        self.num_of_classes = num_of_classes
        self.threshold = threshold
        self.dropout_p = dropout_p
        self.optimizer = optimizer
        self.loss = loss
        self._build_model()

    def _build_model(self):
        # Input layer
        inputs = Input(shape=(self.input_size,), name='sent_input', dtype='int64')

        # Embedding layers
        x = Embedding(self.alphabet_size + 1, self.embedding_size, input_length=self.input_size)(inputs)

        # Convolution layers
        for cl in self.conv_layers:
            x = Convolution1D(cl[0], cl[1])(x)
            x = ThresholdedReLU(self.threshold)(x)
            if not cl[2] is None:
                x = MaxPooling1D(cl[2])(x)
        x = Flatten()(x)

        # Fully connected layers
        for fl in self.fully_connected_layers:
            x = Dense(fl)(x)
            x = ThresholdedReLU(self.threshold)(x)
            x = Dropout(self.dropout_p)(x)

        predictions = Dense(self.num_of_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=predictions)

        model.compile(optimizer=self.optimizer, loss=self.loss)

        self.model = model

    def train(self, training_inputs, training_labels,
              validation_inputs, validation_labels,
              epochs, batch_size):
        self.model.fit(training_inputs, training_labels,
                       validation_data=(validation_inputs, validation_labels),
                       epochs=epochs,
                       batch_size=batch_size)

    def test(self):
        self.model.evaluate()
