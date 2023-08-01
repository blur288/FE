from model import FacialDetectionModel
from face import GetFace
import tensorflow as tf
import numpy as np

def CameraFunction():
    while 1:
        Model.ModelTest()
        Model.LoadModel(filename="Useweights.h5")
        Face = GetFace()
        FaceArray = np.array(Face)
        FaceArray = np.resize(FaceArray, (1, 48, 48, 1))
        Prediction = Model.Predict(FaceArray)
        PredictionList = Prediction.tolist()[0]
        LabelIndex = Prediction.argmax(axis=-1).tolist()[0]
        print(emotionArray[LabelIndex])
        print(Prediction)

emotionArray = ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "suprise"]

if __name__ == "__main__":
    Model = FacialDetectionModel()
    Model.Compile(LearningRate=0.001)

    train = False
    if train == True:
        Model.GetHistory()
    else:
        CameraFunction()



    
    #pip install opencv-python

# # Plot the train and validation loss
# train_loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(train_loss) + 1)
# plt.plot(epochs, train_loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# # Plot the train and validation accuracy
# train_acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# plt.plot(epochs, train_acc, 'bo', label='Training accuracy')
# plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs'x)
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# import seaborn as sns 
# from sklearn.metrics import confusion_matrix

# # Get the true labels and predicted labels for the validation set
# validation_labels = validation_generator.classes
# validation_pred_probs = model.predict(validation_generator)
# validation_pred_labels = np.argmax(validation_pred_probs, axis=1)

# # Compute the confusion matrix
# confusion_mtx = confusion_matrix(validation_labels, validation_pred_labels)
# class_names = list(train_generator.class_indices.keys())
# sns.set()
# sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='YlGnBu', 
#             xticklabels=class_names, yticklabels=class_names)
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')
# plt.show()
