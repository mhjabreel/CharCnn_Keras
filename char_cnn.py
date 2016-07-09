from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Activation
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from keras.layers import Embedding
from keras.layers import ThresholdedReLU
from keras.layers import Dropout
from keras.optimizers import Adam

print "Loading the configurations...",

execfile("config.py")
conv_layers = config.model.conv_layers
fully_layers = config.model.fully_connected_layers
l0 = config.l0
alphabet_size = config.alphabet_size
embedding_size = config.model.embedding_size
num_of_classes = config.num_of_classes
th = config.model.th
p = config.dropout_p
print "Loaded"


print "Building the model...",
# building the model

# Input layer
inputs = Input(shape=(l0,), name='sent_input', dtype='int64')

# Embedding layer

x = Embedding(alphabet_size + 1, embedding_size, input_length=l0)(inputs)

# Convolution layers
for cl in conv_layers:
	x = Convolution1D(cl[0], cl[1])(x)
	x = ThresholdedReLU(th)(x)
	if not cl[2] is None:
		x = MaxPooling1D(cl[2])(x)


x = Flatten()(x)


#Fully connected layers

for fl in fully_layers:
	x = Dense(fl)(x)
	x = ThresholdedReLU(th)(x)
	x = Dropout(0.5)(x)


predictions = Dense(num_of_classes, activation='softmax')(x)

model = Model(input=inputs, output=predictions)

optimizer = Adam()

model.compile(optimizer=optimizer, loss='categorical_crossentropy')

	
print "Built"


print "Loading the data sets...",

from data_utils import Data

train_data = Data(data_source = config.train_data_source, 
                     alphabet = config.alphabet,
                     l0 = config.l0,
                     batch_size = 0,
                     no_of_classes = config.num_of_classes)

train_data.loadData()

X_train, y_train = train_data.getAllData()

dev_data = Data(data_source = config.dev_data_source,
                      alphabet = config.alphabet,
                      l0 = config.l0,
                      batch_size = 0,
                      no_of_classes = config.num_of_classes)


dev_data.loadData()

X_val, y_val = dev_data.getAllData()

print "Loadded"


print "Training ..."

model.fit(X_train, y_train, nb_epoch=config.training.epochs, batch_size=config.batch_size, validation_data=(X_val, y_val))

print "Done!."



