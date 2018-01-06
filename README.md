# FaceRecognitionDemo using dlib and flask
[dlib](http://dlib.net/) is a machine learning library used to detect and recognize the faces

[flask](http://flask.pocoo.org/) is a micro framework to create web page using python

## Instructions
Open cmd prompt, navigate to the directory

Create conda environment with necessary packages using environment.yml file
```
conda env create -f environment.yml
```
Activate the environment
```
conda activate ./environment
```
Run app.py
```
python app.py
```
Open [localhost:5000](http://127.0.0.1:5000/) in the broswer


#### Tutorial followed
* [One](https://towardsdatascience.com/facial-recognition-using-deep-learning-a74e9059a150)
* [Two](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78)

Face recognition can be done in following stages

* Face Detection
* Analyze Facial Feautures
* Compare against known faces
* Make a prediction

#### Face Detection

Face detection is done using Histogram of Oriented Gradients (HOG).In dlib shape_predictor_68_face_landmarks model can be used to get the 68 landmarks of the face.Related code can be found in [face_landmark.py]() file

![68_face_points](https://cdn-images-1.medium.com/max/800/1*AbEg31EgkbXSQehuNJBlWg.png)

#### Facial Features
128 facial measurements (Face embeddings) are genereated by using a pretrained model dlib_face_recognition_resnet_model_v1. Deep learning is used to train the model based on the paper [FaceNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf)

### Comparing the faces
These 128 measurements can be used to recognize the face from previously known faces. A Classifier like SVM can also be trained on these facial embeddings
 


## Packages Used
* dlib 
```
conda install -c conda-forge dlib
```
* Flask
```
conda install Flask
```
* Scikit-image
```
conda install scikit-image
```