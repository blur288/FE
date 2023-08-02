
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import skimage.io
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import DenseNet169

from tensorflow.keras.layers import Dense, Flatten, Dropout,BatchNormalization ,Activation
from tensorflow.keras.models import Model, Sequential
from keras.applications.nasnet import NASNetLarge
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import optimizers
from collections import Counter
import matplotlib.pyplot as plt

from tensorflow.keras.utils import plot_model
from IPython.display import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import skimage.io
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import DenseNet169

from tensorflow.keras.layers import Dense, Flatten, Dropout,BatchNormalization ,Activation
from tensorflow.keras.models import Model, Sequential
from keras.applications.nasnet import NASNetLarge
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import optimizers
from collections import Counter
import matplotlib.pyplot as plt

from tensorflow.keras.utils import plot_model
from IPython.display import Image

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy
import scipy.ndimage
import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf 

from tensorflow.keras.callbacks import ModelCheckpoint

class FacialDetectionModel:

    # /Users/ujjawalprasad/Downloads/archive/test
    train_dir = "../train1" # Directory containing the training data
    test_dir = "../test1"  # Directory containing the validation data


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
        epochs=50,
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
    def SwitchImage(self, imagePath):
        train_datagen = ImageDataGenerator(
            width_shift_range = 0.1,        # Randomly shift the width of images by up to 10%
            height_shift_range = 0.1,       # Randomly shift the height of images by up to 10%
            horizontal_flip = True,         # Flip images horizontally at random
            rescale = 1./255,               # Rescale pixel values to be between 0 and 1
            validation_split = 0.2          # Set aside 20% of the data for validation
        )

        train_generator = train_datagen.flow_from_directory(
            directory = imagePath,           # Directory containing the training data
            target_size = (48, 48),          # Resizes all images to 48x48 pixels
            batch_size = 32,                 # Number of images per batch
            color_mode = "grayscale",        # Converts the images to grayscale
            class_mode = "categorical",      # Classifies the images into 7 categories
            subset = "training"              # Uses the training subset of the data
        )
        return train_generator

class FacialDetectionModel2:

    def initDataGens():
        
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                    validation_split = 0.2,
                                    rotation_range=0.3, #5,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    #zoom_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=False,
                                    fill_mode='nearest')

    #     valid_datagen = ImageDataGenerator(rescale = 1./255,
    #                                   validation_split = 0.2
    #                                   )

        test_datagen  = ImageDataGenerator(rescale = 1./255)


        return train_datagen, test_datagen, test_datagen

    def initDataSets(train, test, train_datagen, valid_datagen, test_datagen, classes_):
        
        train_dataset  = train_datagen.flow_from_directory(directory = '../archive/train1',
                                                    target_size = (48,48),
                                                    class_mode = 'categorical',
                                                    classes=classes_,
                                                    subset = 'training',
                                                    batch_size = 64)

        valid_dataset = train_datagen.flow_from_directory(directory = '../archive/train1',
                                                    target_size = (48,48),
                                                    class_mode = 'categorical',
                                                    classes=classes_,
                                                    subset = 'validation',
                                                    batch_size = 64)
        
        
    #     valid_dataset = valid_datagen.flow_from_directory(directory = train,#'../input/fer2013/train',
    #                                                   target_size = (48,48),
    #                                                   class_mode = 'categorical',
    #                                                   classes=classes_,
    #                                                   subset = 'validation',
    #                                                   batch_size = 64)

        test_dataset = test_datagen.flow_from_directory(directory = '../archive/test1',
                                                    target_size = (48,48),
                                                    class_mode = 'categorical',
                                                    classes=classes_,
                                                    batch_size = 64)


        return train_dataset, valid_dataset, test_dataset

    def updateBaseModel(base_model, num_classes):
        
        for layer in base_model.layers[:]:
            layer.trainable=True
        
        # Building Model
        model=Sequential()
        model.add(base_model)
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(32,kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32,kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32,kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(num_classes, activation='softmax'))

        return model
        
    #     df = pd.DataFrame(l, columns=['index', 'observers'])
    #     print(df)
    # ['Anger', 'Disgust', 'Fear', 'Happy','Neutral','Sad', 'Surprise']

    def getClassWeights(train_dataset):
        
        counter = Counter(train_dataset.classes)                          
        max_val = float(max(counter.values()))       
        # class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}  
        class_weights = {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1}  
        print(class_weights)

        return class_weights

    def getAllForTraining(model, save_h5_to_path, epochs_):
        
        lrd = ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1,
        )

        mcp_5categories = ModelCheckpoint(save_h5_to_path) 

        # es = EarlyStopping(verbose=1, patience=20)
        es = EarlyStopping(
            monitor='val_accuracy',
            min_delta=0.00005,
            patience=11,
            verbose=1,
            restore_best_weights=True,
        )

        # optimizers.Adam(learning_rate=1e-3, decay=1e-3 / epochs)
        # model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=METRICS)
        t_epochs = epochs_

        optim = optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer=optim, loss='categorical_crossentropy',metrics=METRICS)
        model.summary()

        return lrd, mcp_5categories, es, t_epochs, model, t_epochs

    # def plotConfusionMatrix(model, test_dataset, num_of_test_samples, batch_size, target_names):
        
    #     #Confution Matrix and Classification Report
    #     Y_pred = model.predict(test_dataset, num_of_test_samples // batch_size+1)
    #     y_pred = np.argmax(Y_pred, axis=1)
    #     print('Confusion Matrix')
    #     cm = confusion_matrix(test_dataset.classes, y_pred)
    #     print('Classification Report')
    #     print(classification_report(test_dataset.classes, y_pred, target_names=target_names))
    #     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    #     disp.plot(cmap=plt.cm.Blues)
    #     plt.show()

    target_names = ['Anger','Disgust','Fear','Happy','Neutral','Sad','Surprise']
    num_classes = len(target_names)
    train_path = '../archive/train1'
    test_path = '../archive/test1'
    save_model_h5_to_path = 'realfinalfrweightsusethisoneplz.h5'
    epochs = 25

    train_datagen, valid_datagen, test_datagen = initDataGens()
    train_dataset, valid_dataset, test_dataset = initDataSets(train_path, test_path, train_datagen, valid_datagen, test_datagen, target_names)
    class_weights = getClassWeights(train_dataset)

    # base_model = tf.keras.applications.ResNet50(input_shape=(48,48,3),include_top=False,weights="imagenet")
    base_model = tf.keras.applications.MobileNet(input_shape=(48,48,3),include_top=False,weights="imagenet")

    # model = build_net(optimizers.SGD(learning_rate=0.01, momentum=0.9), 7, METRICS)

    model = updateBaseModel(base_model, len(target_names))
    lrd, mcp_5categories, es, t_epochs, model, t_epochs = getAllForTraining(model, save_model_h5_to_path, epochs)

    history1 = model.fit(train_dataset,validation_data=valid_dataset,epochs = t_epochs,verbose = 1,callbacks=[lrd,mcp_5categories,es], class_weight=class_weights)