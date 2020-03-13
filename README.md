# facenet-cs496
alex cao's implementation of facenet on pubfig83 for cs 496 (winter 2020 with prof. liu)

# project description
the problem this project is targeting is facial recognition of a library of people. so given a library of identities, we'd like to classify a test face photo as one of the given identities or unknown. this project solves this problem by implementing a version of the facenet model (https://arxiv.org/abs/1503.03832) combined with thresholding and nearest centroid on the embedding.

i train my version of the facenet model (smaller cnn network, same triplet loss) on the first 63 identities from the pubfig83 dataset (each identity has over 100 face photos). i reserve 20 photos from each of those identities for testing. i test my model on all 83 identities (20 new identities with all photos and 63 known/trained identities with reserved testing photos). i compute the centroid for each training identity's embedding. for classification, if the distance to the nearest centroid is greater than some threshold, classify as unknown, otherwise classify as nearest centroid's identity.

input: pubfig83 dataset testing split, library of 83 people with multiple photos each  
output: (i) testing only on 63 known identities, testing.py computes top-1 and top-6 accuracy as well as respective guesses and (ii) testing on all  identities, testing.py computes top-1 accuracies (63 identities plus unknown class) for a threshold sweep

# required python packages
python packages my code uses:

numpy  
tensorflow  
os  
glob  
pylab  
sklearn  
random  

i ran some of my python code on my local laptop (spyder) and on some on my research lab's gpu servers (mainly training model). my code is basically a straightforward use of tensorflow and numpy so there are no important dependencies, etc. but for completeness i will list python environments/packages of each. 

on my local laptop i simply use the spyder base/root environment from anaconda (updated 2019). on the gpu servers, the docker image i use can be found on the docker hub as dklabjan/keras-gpu. it contains the following packages:

Package              Version               
-------------------- ----------------------
absl-py              0.7.0                 
astor                0.7.1                 
boto                 2.49.0                
boto3                1.9.134               
botocore             1.12.134              
bz2file              0.98                  
certifi              2019.3.9              
chardet              3.0.4                 
cycler               0.10.0                
docutils             0.14                  
gast                 0.2.2                 
gensim               3.7.2                 
grpcio               1.19.0                
h5py                 2.9.0                 
idna                 2.8                   
imbalanced-learn     0.4.3                 
jmespath             0.9.4                 
Keras                2.2.4                 
Keras-Applications   1.0.7                 
Keras-Preprocessing  1.0.9                 
kiwisolver           1.0.1                 
Markdown             3.0.1                 
matplotlib           3.0.3                 
mock                 2.0.0                 
nltk                 3.4.1                 
numpy                1.16.2                
pandas               0.24.2                
pbr                  5.1.2                 
pip                  19.0.3                
protobuf             3.6.1                 
pycurl               7.43.0                
pygobject            3.20.0                
pyparsing            2.4.0                 
python-apt           1.1.0b1+ubuntu0.16.4.2   
python-dateutil      2.8.0                 
pytz                 2019.1                
PyYAML               5.1                   
requests             2.21.0                
s3transfer           0.2.0                 
scikit-learn         0.20.3                
scipy                1.2.1                 
seaborn              0.9.0                 
setuptools           40.8.0                
six                  1.12.0                
smart-open           1.8.2                 
tensorboard          1.13.0                
tensorflow-estimator 1.13.0                
tensorflow-gpu       1.13.1                
termcolor            1.1.0                 
urllib3              1.24.2                
Werkzeug             0.14.1                
wheel                0.29.0 

please note that all python code should run on both anaconda root environment (spyder) on local computer and docker image "dklabjan/keras-gpu".

# trained model
see "trained-model" directory. "model-blueprint.txt" contains layer-by-layer description of deep ccn architecture used in my facenet implementation. i saved each layer's weights (w) and biases (b) as .npy files. for instance, "w5.npy" contains the weights for the 5th variable layer (not including max pools, normalizations, etc.)

see "transform-images_npy-to-embeddings.py" in "use-trained-model" directory for example on how to load saved  weights/biases to  use model.  i realize this is not as clean as a single model file but it was much faster to load than tensorflow's save graph/model method

# how to train the model from scratch
see "train-model-from-scratch" directory

to train, simply run "train.py". this trains the model (see "model-blueprint.txt" in "trained-model" directory) for 100,000 iterations of mini-batch size 128 using the triplet loss. the training split of pubfig83 identities/images is in "/recognizer/pubfig83-train". it prints 2 outputs from our generator just to see that it works and an example triplet of images. finally it saves all the weight and biases at the end of training. see trained model section of this readme

the "recognizer" directory contains the meat of model training. the training and test splits of the dataset people/images are in their respective directories. "model.py" defines the cnn model. "tf_dataset.py" builds the tensorflow data input pipeline. "triplet_generator.py" builds the iterator which generates random samples of triplets for training

** be careful that no ".DS_Store" folders appear in ""/recognizer/pubfig83-train" or "/recognizer/pubfig83-test" as they will cause error with dictionary building. i don't know what they are but sometimes appear randomly in my github

# how to use trained model
see "use-trained-model" directory. many of the files are copied over from "train-model-from-scratch" so the user who downloads repository doesn't have to move things or redefine paths

1. first run "people-to-class_num-dictionary.py" to create a .npy  file holding the dictionary between celebrity name/identity and class number i.e. 0-63. the output is already created in the github repository

2. run "convert-jpg_images-to-npy.py" to convert all jpg images of people into .npy arrays and concatenate them into single .npy file. i know this must be horrible practice but it was the easiest to implement to feed into tensorflow model. this almost certainly needs to be run on a gpu server for speed. the output of this code will be 4 files:  
(i) train-images.npy - (9295, 128, 128, 3) array of  all training images  
(ii) train-labels.npy - (9295) array of all training image labels  
(iii) test-images.npy - (4543, 128, 128, 3) array of  all test images  
(iv) test-labels.npy - (4543) array of all test image labels  
the image .npy files are not copied into the  github repository for size reasons. but they are not strictly needed since i do save the embeddings of all images - see next line

3. run "transform-images_npy-to-embeddings.py" but this assumes you've completed above step (2). this script runs all images (.npy format) through tensorflow model loaded from .npy weights/biases files and saves respective 128-dimensional embeddings in same format i.e. (9295, 128) array, etc.  the output is: "train-embeddings.npy" and "test-embeddings.npy"

4. run "plot-embeddings.py" to visualize training/testing embeddings. outputs:  
(i) tsne of training data with corresponding triplet of images   
(ii) k-means sse plot for training data  
(iii) tsne of "known" testing data with training centroids  
(iv) tsne of "known" and "unknown" testing data with training centroids

5. run "test-embeddings.npy". see output description under project description section
