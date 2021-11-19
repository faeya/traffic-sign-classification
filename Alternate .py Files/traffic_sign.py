#this has been tested on my MacBook Pro using Jupyter Notebooks, if running on Windows or other OS, please ensure the necessary system call changes are made.
#if running on local system please run the following command to install prerequisites (packages) :
#pip install tensorflow keras sklearn matplotlib pandas pillow opencv-python
#importing required modules to create this project.
import numpy as np  #basic exploration and assist.
import pandas as pd #basic exploration and assist.
import matplotlib.pyplot as plt #to plot graphs.
import cv2 #to classify image and assist recognition.
import tensorflow as tf
from PIL import Image #to create image arrays etc.
import os #to allow operating system calls.
from sklearn.model_selection import train_test_split #scikit learn to split, test and train the data.
from tensorflow.keras.utils import to_categorical #to encode data to categories.
from keras.models import Sequential, load_model #to create our model which is sequential.
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout #to add layers to our model for accurate results.

data = []
labels = []
classes = 43
cur_path = os.getcwd()

#retrieving the images and their labels and appending said items in previously initialised lists.
for i in range(classes):
    path = os.path.join(cur_path,'train',str(i))
    images = os.listdir(path)
    for a in images:
        try:
            image = Image.open(path + '/'+ a)
            image = image.resize((30,30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")

#converting lists into numpy arrays to feed model.
data = np.array(data)
labels = np.array(labels)
print(data.shape, labels.shape) #The shape of data obtained is (39209, 30, 30, 3) which means that there are 39,209 images of size 30×30 pixels and the last 3 means the data contains colored images (RGB value)

#splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#converting the labels into one-hot encoding of categories.
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

#building the CNN model as CNN is best for image classification purposes.
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:])) #2D convolution layer class that creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2))) #max pooling operation for 2D spatial data.
model.add(Dropout(rate=0.25)) #applies dropout layer that randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting.
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten()) #flattens the input without affecting the batch size.
model.add(Dense(256, activation='relu')) #regular densely-connected NN layer.
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

#compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])#we compile the model with Adam optimizer which performs well and loss is “categorical_crossentropy” because we have multiple classes to categorise.
epochs = 15 #we tried with batch size 32 and 64. our model performed better with 64 batch size and after 15 epochs the accuracy was stable.
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
model.save("my_model.h5")

#with matplotlib, we plot the graph for accuracy and the loss.
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

#testing accuracy on test dataset
from sklearn.metrics import accuracy_score
y_test = pd.read_csv('Test.csv')
labels = y_test["ClassId"].values
imgs = y_test["Path"].values
data=[]
for img in imgs:
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))
X_test=np.array(data)
predict_x=model.predict(X_test) 
classes_x=np.argmax(predict_x,axis=1)

#accuracy with the test data
from sklearn.metrics import accuracy_score
print(accuracy_score(labels, pred)) #our model got a 95% accuracy.
