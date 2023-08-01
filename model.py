from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy
import scipy.ndimage
import tensorflow as tf

# /Users/ujjawalprasad/Downloads/archive/test
train_dir = "./fer2013/train" # Directory containing the training data
test_dir = "./fer2013/test"  # Directory containing the validation data


train_datagen = ImageDataGenerator(
    width_shift_range = 0.1,        # Randomly shift the width of images by up to 10%
    height_shift_range = 0.1,       # Randomly shift the height of images by up to 10%
    horizontal_flip = True,         # Flip images horizontally at random
    rescale = 1./255,               # Rescale pixel values to be between 0 and 1
    validation_split = 0.2          # Set aside 20% of the data for validation
)

validation_datagen = ImageDataGenerator(
    rescale = 1./255,               # Rescale pixel values to be between 0 and 1
    validation_split = 0.2          # Set aside 20% of the data for validation
)

train_generator = train_datagen.flow_from_directory(
    directory = train_dir,           # Directory containing the training data
    target_size = (48, 48),          # Resizes all images to 48x48 pixels
    batch_size = 32,                 # Number of images per batch
    color_mode = "grayscale",        # Converts the images to grayscale
    class_mode = "categorical",      # Classifies the images into 7 categories
    subset = "training"              # Uses the training subset of the data
)

validation_generator = validation_datagen.flow_from_directory(
    directory = test_dir,            # Directory containing the validation data
    target_size = (48, 48),          # Resizes all images to 48x48 pixels
    batch_size = 32,                 # Number of images per batch
    color_mode = "grayscale",        # Converts the images to grayscale
    class_mode = "categorical",      # Classifies the images into 7 categories
    subset = "validation"            # Uses the validation subset of the data
)

from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf 

from tensorflow.keras.callbacks import ModelCheckpoint

class FacialDetectionModel:
    model = Sequential()
    def __init__(self) -> None:
        # Add a convolutional layer with 32 filters, 3x3 kernel size, and relu activation function
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        # Add a batch normalization layer
        self.model.add(BatchNormalization())
        # Add a second convolutional layer with 64 filters, 3x3 kernel size, and relu activation function
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        # Add a second batch normalization layer
        self.model.add(BatchNormalization())
        # Add a max pooling layer with 2x2 pool size
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # Add a dropout layer with 0.25 dropout rate
        self.model.add(Dropout(0.25))
        # Add a third convolutional layer with 128 filters, 3x3 kernel size, and relu activation function
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        # Add a third batch normalization layer
        self.model.add(BatchNormalization())
        # Add a fourth convolutional layer with 128 filters, 3x3 kernel size, and relu activation function
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        # Add a fourth batch normalization layer
        self.model.add(BatchNormalization())
        # Add a max pooling layer with 2x2 pool size
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # Add a dropout layer with 0.25 dropout rate
        self.model.add(Dropout(0.25))
        # Add a fifth convolutional layer with 256 filters, 3x3 kernel size, and relu activation function
        self.model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
        # Add a fifth batch normalization layer
        self.model.add(BatchNormalization())
        # Add a sixth convolutional layer with 256 filters, 3x3 kernel size, and relu activation function
        self.model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
        # Add a sixth batch normalization layer
        self.model.add(BatchNormalization())
        # Add a max pooling layer with 2x2 pool size
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # Add a dropout layer with 0.25 dropout rate
        self.model.add(Dropout(0.25))
        # Flatten the output of the convolutional layers
        self.model.add(Flatten())
        # Add a dense layer with 256 neurons and relu activation function
        self.model.add(Dense(256, activation='relu'))
        # Add a seventh batch normalization layer
        self.model.add(BatchNormalization())
        # Add a dropout layer with 0.5 dropout rate
        self.model.add(Dropout(0.5))
        # Add a dense layer with 7 neurons (one for each class) and softmax activation function
        self.model.add(Dense(8, activation='softmax'))
    def Compile(self, LearningRate):
        # Compile the model with categorical cross-entropy loss, adam optimizer, and accuracy metric
        self.model.compile(loss="categorical_crossentropy", optimizer= tf.keras.optimizers.Adam(lr=LearningRate), metrics=['accuracy'])
        self.model.compile(loss="categorical_crossentropy", optimizer= tf.keras.optimizers.Adam(learning_rate=LearningRate), metrics=['accuracy'])
    def GetHistory(self, filename="model_weights.h5"):
        checkpoint_callback = ModelCheckpoint(
            filepath=filename,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            verbose=1
        )
        
        history = self.model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=40,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint_callback]
        )

    def LoadModel(self, filename="weights.h5"):
        self.model.load_weights(filename)
    def ModelTest(self):
        results = self.model.evaluate(validation_generator)
        print(results)
    def Predict(self, image):
        prediciton = self.model.predict(image)
        return prediciton
