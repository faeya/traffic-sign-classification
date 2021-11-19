# Traffic Sign Classification
### A Deep Learning Project for Traffic Sign Classification

This project is an approach to recognise and classify traffic signs, an important part of self-driving cars as to achieve level 5 autonomous, it is necessary for vehicles to understand and follow all traffic rules. So, for achieving accuracy in this technology, the vehicles should be able to interpret traffic signs and make decisions accordingly.

There are several different types of traffic signs like speed limits, no entry, traffic signals, turn left or right, children crossing, no passing of heavy vehicles, etc. Traffic signs classification is the process of identifying which class a traffic sign belongs to.

In this Python project, we have built a deep neural network model that can classify traffic signs present in the image into different categories. With this model, we are able to read and understand traffic signs which are a very important task for all autonomous vehicles.

For this project, we are using the public Kaggle dataset available at  : https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

The dataset contains more than 50,000 images of different traffic signs. It is further classified into 43 different classes. The dataset is quite varying, some of the classes have many images while some classes have few images. The dataset has a train folder which contains images inside each class and a test folder which we will use for testing our model.

> Our model has been successfully tested in Jupyter Notebooks, the file of which has been uploaded named : DL_TrafficSignClassification_MP.ipynb.

> The code execution flow on my system can also be viewed by acessing this file uploaded named : DL_TrafficSignClassification_MP.html

> In case there is an issue with the .ipnb file, I have uplaoded the code converted to .py separately in the 'Alternative .py Files' folder.

> The model file 'traffic_classifier.h5' is created by running the traffic_sign.py file. Execute the traffic_sign.py file on your local system to create this model from scratch if required.

*Please download the files on your local device and make sure that the dataset's .csv files and image folders are present in the same directory as the .py / .ipynb file for correct code execution.*
