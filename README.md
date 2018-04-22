# All the work is done in intro_to_tensorflow.ipynb, below is the .md output 

<h1 align="center">TensorFlow Neural Network Lab</h1>

<img src="image/notmnist.png">
In this lab, you'll use all the tools you learned from *Introduction to TensorFlow* to label images of English letters! The data you are using, <a href="http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html">notMNIST</a>, consists of images of a letter from A to J in different fonts.

The above images are a few examples of the data you'll be training on. After training the network, you will compare your prediction model against test data. Your goal, by the end of this lab, is to make predictions against that test set with at least an 80% accuracy. Let's jump in!

To start this lab, you first need to import all the necessary modules. Run the code below. If it runs successfully, it will print "`All modules imported`".


```python
import hashlib
import os
import pickle
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from tqdm import tqdm
from zipfile import ZipFile

print('All modules imported.')
```

    All modules imported.
    

The notMNIST dataset is too large for many computers to handle.  It contains 500,000 images for just training.  You'll be using a subset of this data, 15,000 images for each label (A-J).


```python
def download(url, file):
    """
    Download file from <url>
    :param url: URL to file
    :param file: Local file path
    """
    if not os.path.isfile(file):
        print('Downloading ' + file + '...')
        urlretrieve(url, file)
        print('Download Finished')

# Download the training and test dataset.
download('https://s3.amazonaws.com/udacity-sdc/notMNIST_train.zip', 'notMNIST_train.zip')
download('https://s3.amazonaws.com/udacity-sdc/notMNIST_test.zip', 'notMNIST_test.zip')

# Make sure the files aren't corrupted
assert hashlib.md5(open('notMNIST_train.zip', 'rb').read()).hexdigest() == 'c8673b3f28f489e9cdf3a3d74e2ac8fa',\
        'notMNIST_train.zip file is corrupted.  Remove the file and try again.'
assert hashlib.md5(open('notMNIST_test.zip', 'rb').read()).hexdigest() == '5d3c7e653e63471c88df796156a9dfa9',\
        'notMNIST_test.zip file is corrupted.  Remove the file and try again.'

# Wait until you see that all files have been downloaded.
print('All files downloaded.')
```

    All files downloaded.
    


```python
def uncompress_features_labels(file):
    """
    Uncompress features and labels from a zip file
    :param file: The zip file to extract the data from
    """
    features = []
    labels = []

    with ZipFile(file) as zipf:
        # Progress Bar
        filenames_pbar = tqdm(zipf.namelist(), unit='files')
        
        # Get features and labels from all files
        for filename in filenames_pbar:
            # Check if the file is a directory
            if not filename.endswith('/'):
                with zipf.open(filename) as image_file:
                    image = Image.open(image_file)
                    image.load()
                    # Load image data as 1 dimensional array
                    # We're using float32 to save on memory space
                    feature = np.array(image, dtype=np.float32).flatten()

                # Get the the letter from the filename.  This is the letter of the image.
                label = os.path.split(filename)[1][0]

                features.append(feature)
                labels.append(label)
    return np.array(features), np.array(labels)

# Get the features and labels from the zip files
train_features, train_labels = uncompress_features_labels('notMNIST_train.zip')
test_features, test_labels = uncompress_features_labels('notMNIST_test.zip')

# Limit the amount of data to work with a docker container
docker_size_limit = 150000
train_features, train_labels = resample(train_features, train_labels, n_samples=docker_size_limit)

# Set flags for feature engineering.  This will prevent you from skipping an important step.
is_features_normal = False
is_labels_encod = False

# Wait until you see that all features and labels have been uncompressed.
print('All features and labels uncompressed.')
```

    100%|█████████████████████████████████| 210001/210001 [00:46<00:00, 4523.69files/s]
    100%|███████████████████████████████████| 10001/10001 [00:02<00:00, 4638.89files/s]
    

    All features and labels uncompressed.
    

<img src="image/Mean_Variance_Image.png" style="height: 75%;width: 75%; position: relative; right: 5%">
## Problem 1
The first problem involves normalizing the features for your training and test data.

Implement Min-Max scaling in the `normalize_grayscale()` function to a range of `a=0.1` and `b=0.9`. After scaling, the values of the pixels in the input data should range from 0.1 to 0.9.

Since the raw notMNIST image data is in [grayscale](https://en.wikipedia.org/wiki/Grayscale), the current values range from a min of 0 to a max of 255.

Min-Max Scaling:
$
X'=a+{\frac {\left(X-X_{\min }\right)\left(b-a\right)}{X_{\max }-X_{\min }}}
$

*If you're having trouble solving problem 1, you can view the solution [here](https://github.com/udacity/deep-learning/blob/master/intro-to-tensorflow/intro_to_tensorflow_solution.ipynb).*


```python
# Problem 1 - Implement Min-Max scaling for grayscale image data
def normalize_grayscale(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    # TODO: Implement Min-Max scaling for grayscale image data
    X = image_data
    max_ = 0.9
    min_ = 0.1
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (max_ - min_) + min_
    return X_scaled


### DON'T MODIFY ANYTHING BELOW ###
# Test Cases
np.testing.assert_array_almost_equal(
    normalize_grayscale(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255])),
    [0.1, 0.103137254902, 0.106274509804, 0.109411764706, 0.112549019608, 0.11568627451, 0.118823529412, 0.121960784314,
     0.125098039216, 0.128235294118, 0.13137254902, 0.9], 
    decimal=3)
np.testing.assert_array_almost_equal(
    normalize_grayscale(np.array([0, 1, 10, 20, 30, 40, 233, 244, 254,255])),
    [0.1, 0.103137254902, 0.13137254902, 0.162745098039, 0.194117647059, 0.225490196078, 0.830980392157, 0.865490196078,
     0.896862745098, 0.9])

if not is_features_normal:
    train_features = normalize_grayscale(train_features)
    test_features = normalize_grayscale(test_features)
    is_features_normal = True

print('Tests Passed!')
```

    Tests Passed!
    


```python
if not is_labels_encod:
    # Turn labels into numbers and apply One-Hot Encoding
    encoder = LabelBinarizer()
    encoder.fit(train_labels)
    train_labels = encoder.transform(train_labels)
    test_labels = encoder.transform(test_labels)

    # Change to float32, so it can be multiplied against the features in TensorFlow, which are float32
    train_labels = train_labels.astype(np.float32)
    test_labels = test_labels.astype(np.float32)
    is_labels_encod = True

print('Labels One-Hot Encoded')
```

    Labels One-Hot Encoded
    


```python
assert is_features_normal, 'You skipped the step to normalize the features'
assert is_labels_encod, 'You skipped the step to One-Hot Encode the labels'

# Get randomized datasets for training and validation
train_features, valid_features, train_labels, valid_labels = train_test_split(
    train_features,
    train_labels,
    test_size=0.05,
    random_state=832289)

print('Training features and labels randomized and split.')
```

    Training features and labels randomized and split.
    


```python
# Save the data for easy access
pickle_file = 'notMNIST.pickle'
if not os.path.isfile(pickle_file):
    print('Saving data to pickle file...')
    try:
        with open('notMNIST.pickle', 'wb') as pfile:
            pickle.dump(
                {
                    'train_dataset': train_features,
                    'train_labels': train_labels,
                    'valid_dataset': valid_features,
                    'valid_labels': valid_labels,
                    'test_dataset': test_features,
                    'test_labels': test_labels,
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

print('Data cached in pickle file.')
```

    Data cached in pickle file.
    

# Checkpoint
All your progress is now saved to the pickle file.  If you need to leave and comeback to this lab, you no longer have to start from the beginning.  Just run the code block below and it will load all the data and modules required to proceed.


```python
%matplotlib inline

# Load the modules
import pickle
import math

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

# Reload the data
pickle_file = 'notMNIST.pickle'
with open(pickle_file, 'rb') as f:
    pickle_data = pickle.load(f)
    train_features = pickle_data['train_dataset']
    train_labels = pickle_data['train_labels']
    valid_features = pickle_data['valid_dataset']
    valid_labels = pickle_data['valid_labels']
    test_features = pickle_data['test_dataset']
    test_labels = pickle_data['test_labels']
    del pickle_data  # Free up memory

print('Data and modules loaded.')
```

    Data and modules loaded.
    


## Problem 2

Now it's time to build a simple neural network using TensorFlow. Here, your network will be just an input layer and an output layer.

<img src="image/network_diagram.png" style="height: 40%;width: 40%; position: relative; right: 10%">

For the input here the images have been flattened into a vector of $28 \times 28 = 784$ features. Then, we're trying to predict the image digit so there are 10 output units, one for each label. Of course, feel free to add hidden layers if you want, but this notebook is built to guide you through a single layer network. 

For the neural network to train on your data, you need the following <a href="https://www.tensorflow.org/resources/dims_types.html#data-types">float32</a> tensors:
 - `features`
  - Placeholder tensor for feature data (`train_features`/`valid_features`/`test_features`)
 - `labels`
  - Placeholder tensor for label data (`train_labels`/`valid_labels`/`test_labels`)
 - `weights`
  - Variable Tensor with random numbers from a truncated normal distribution.
    - See <a href="https://www.tensorflow.org/api_docs/python/constant_op.html#truncated_normal">`tf.truncated_normal()` documentation</a> for help.
 - `biases`
  - Variable Tensor with all zeros.
    - See <a href="https://www.tensorflow.org/api_docs/python/constant_op.html#zeros"> `tf.zeros()` documentation</a> for help.

*If you're having trouble solving problem 2, review "TensorFlow Linear Function" section of the class.  If that doesn't help, the solution for this problem is available [here](intro_to_tensorflow_solution.ipynb).*


```python
# All the pixels in the image (28 * 28 = 784)
features_count = 784
# All the labels
labels_count = 10

# TODO: Set the features and labels tensors
# placeholder
features = tf.placeholder("float", [None, features_count])
labels = tf.placeholder("float", [None, labels_count])

# TODO: Set the weights and biases tensors
weights = tf.Variable(tf.truncated_normal( # Outputs random values from a truncated normal distribution.
                                            [features_count, labels_count], #
                                            stddev=1.0 / math.sqrt(float(features_count)))
                      , name='weights')
biases = tf.Variable(tf.zeros([labels_count]), name='biases')



### DON'T MODIFY ANYTHING BELOW ###

#Test Cases
from tensorflow.python.ops.variables import Variable

assert features._op.name.startswith('Placeholder'), 'features must be a placeholder'
assert labels._op.name.startswith('Placeholder'), 'labels must be a placeholder'
assert isinstance(weights, Variable), 'weights must be a TensorFlow variable'
assert isinstance(biases, Variable), 'biases must be a TensorFlow variable'

assert features._shape == None or (\
    features._shape.dims[0].value is None and\
    features._shape.dims[1].value in [None, 784]), 'The shape of features is incorrect'
assert labels._shape  == None or (\
    labels._shape.dims[0].value is None and\
    labels._shape.dims[1].value in [None, 10]), 'The shape of labels is incorrect'
assert weights._variable._shape == (784, 10), 'The shape of weights is incorrect'
assert biases._variable._shape == (10), 'The shape of biases is incorrect'

assert features._dtype == tf.float32, 'features must be type float32'
assert labels._dtype == tf.float32, 'labels must be type float32'

# Feed dicts for training, validation, and test session
train_feed_dict = {features: train_features, labels: train_labels}
valid_feed_dict = {features: valid_features, labels: valid_labels}
test_feed_dict = {features: test_features, labels: test_labels}

# Linear Function WX + b
logits = tf.matmul(features, weights) + biases

prediction = tf.nn.softmax(logits)

# Cross entropy
cross_entropy = - tf.reduce_sum(labels * tf.log(prediction), reduction_indices=1)
# tf.log = Computes natural logarithm of x element-wise.
# tf.reduce_sum = Computes the sum of elements across dimensions of a tensor. (deprecated arguments)
#      reduction_indices: The old (deprecated) name for axis.
# x = tf.constant([[1, 1, 1], [1, 1, 1]])
# tf.reduce_sum(x)  # 6
# tf.reduce_sum(x, 0)  # [2, 2, 2]
# tf.reduce_sum(x, 1)  # [3, 3]

# Training loss
loss = tf.reduce_mean(cross_entropy)
# tf.reduce_mean = Computes the mean of elements across dimensions of a tensor. (deprecated arguments)
# x = tf.constant([[1., 1.], [2., 2.]])
# tf.reduce_mean(x)  # 1.5
# tf.reduce_mean(x, 0)  # [1.5, 1.5]
# tf.reduce_mean(x, 1)  # [1.,  2.]

# Create an operation that initializes all variables
init = tf.global_variables_initializer()

# Test Cases
with tf.Session() as session:
    session.run(init)
    session.run(loss, feed_dict=train_feed_dict)
    session.run(loss, feed_dict=valid_feed_dict)
    session.run(loss, feed_dict=test_feed_dict)
    biases_data = session.run(biases)

assert not np.count_nonzero(biases_data), 'biases must be zeros'

print('Tests Passed!')
```

    WARNING:tensorflow:Tensor._shape is private, use Tensor.shape instead. Tensor._shape will eventually be removed.
    WARNING:tensorflow:Tensor._shape is private, use Tensor.shape instead. Tensor._shape will eventually be removed.
    WARNING:tensorflow:Tensor._shape is private, use Tensor.shape instead. Tensor._shape will eventually be removed.
    WARNING:tensorflow:Tensor._shape is private, use Tensor.shape instead. Tensor._shape will eventually be removed.
    WARNING:tensorflow:Tensor._shape is private, use Tensor.shape instead. Tensor._shape will eventually be removed.
    WARNING:tensorflow:Tensor._shape is private, use Tensor.shape instead. Tensor._shape will eventually be removed.
    WARNING:tensorflow:Tensor._shape is private, use Tensor.shape instead. Tensor._shape will eventually be removed.
    WARNING:tensorflow:Tensor._shape is private, use Tensor.shape instead. Tensor._shape will eventually be removed.
    Tests Passed!
    


```python
# Determine if the predictions are correct
is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
# Calculate the accuracy of the predictions
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

print('Accuracy function created.')
```

    Accuracy function created.
    

<img src="image/Learn_Rate_Tune_Image.png" style="height: 70%;width: 70%">
## Problem 3
Below are 2 parameter configurations for training the neural network. In each configuration, one of the parameters has multiple options. For each configuration, choose the option that gives the best acccuracy.

Parameter configurations:

Configuration 1
* **Epochs:** 1
* **Learning Rate:**
  * 0.8
  * 0.5
  * 0.1
  * 0.05
  * 0.01

Configuration 2
* **Epochs:**
  * 1
  * 2
  * 3
  * 4
  * 5
* **Learning Rate:** 0.2

The code will print out a Loss and Accuracy graph, so you can see how well the neural network performed.

*If you're having trouble solving problem 3, you can view the solution [here](intro_to_tensorflow_solution.ipynb).*


```python
def testing_config(batch_size, epochs, learning_rate):
    ### DON'T MODIFY ANYTHING BELOW ###
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)    

    # The accuracy measured against the validation set
    validation_accuracy = 0.0

    # Measurements use for graphing loss and accuracy
    log_batch_step = 50
    batches = []
    loss_batch = []
    train_acc_batch = []
    valid_acc_batch = []

    with tf.Session() as session:
        session.run(init)
        batch_count = int(math.ceil(len(train_features)/batch_size))

        for epoch_i in range(epochs):

            # Progress bar
            batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')

            # The training cycle
            for batch_i in batches_pbar:
                # Get a batch of training features and labels
                batch_start = batch_i*batch_size
                batch_features = train_features[batch_start:batch_start + batch_size]
                batch_labels = train_labels[batch_start:batch_start + batch_size]

                # Run optimizer and get loss
                _, l = session.run(
                    [optimizer, loss],
                    feed_dict={features: batch_features, labels: batch_labels})

                # Log every 50 batches
                if not batch_i % log_batch_step:
                    # Calculate Training and Validation accuracy
                    training_accuracy = session.run(accuracy, feed_dict=train_feed_dict)
                    validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

                    # Log batches
                    previous_batch = batches[-1] if batches else 0
                    batches.append(log_batch_step + previous_batch)
                    loss_batch.append(l)
                    train_acc_batch.append(training_accuracy)
                    valid_acc_batch.append(validation_accuracy)

            # Check accuracy against Validation data
            validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

    loss_plot = plt.subplot(211)
    loss_plot.set_title('Loss')
    loss_plot.plot(batches, loss_batch, 'g')
    loss_plot.set_xlim([batches[0], batches[-1]])
    acc_plot = plt.subplot(212)
    acc_plot.set_title('Accuracy')
    acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
    acc_plot.plot(batches, valid_acc_batch, 'x', label='Validation Accuracy')
    acc_plot.set_ylim([0, 1.0])
    acc_plot.set_xlim([batches[0], batches[-1]])
    acc_plot.legend(loc=4)
    plt.tight_layout()
    plt.show()

    print('Validation accuracy at {}'.format(validation_accuracy))

    
    
# Change if you have memory restrictions
batch_size = 128

# TODO: Find the best parameters for each configuration
epochs = [[1] * 5, range(1,6)]
learning_rate = [[0.8, 0.5, 0.1, 0.05, 0.01], [0.2] * 5]

for e, lr in zip(epochs, learning_rate):
    for e_i, lr_i in zip(e, lr):
        print('epochs ={}, learning_rate ={}'.format(e_i, lr_i))
        testing_config(batch_size, e_i, lr_i)
```

    epochs =1, learning_rate =0.8
    

    
    Epoch  1/1:   0%|                                                                                                                                                                                   | 0/1114 [00:00<?, ?batches/s]
    Epoch  1/1:   0%|▏                                                                                                                                                                          | 1/1114 [00:00<08:59,  2.06batches/s]
    Epoch  1/1:   2%|███▉                                                                                                                                                                      | 26/1114 [00:00<00:24, 43.97batches/s]
    Epoch  1/1:   4%|███████▎                                                                                                                                                                  | 48/1114 [00:00<00:15, 68.39batches/s]
    Epoch  1/1:   5%|█████████                                                                                                                                                                 | 59/1114 [00:01<00:21, 49.10batches/s]
    Epoch  1/1:   7%|████████████▎                                                                                                                                                             | 81/1114 [00:01<00:16, 61.63batches/s]
    Epoch  1/1:   9%|███████████████▎                                                                                                                                                         | 101/1114 [00:01<00:18, 55.00batches/s]
    Epoch  1/1:  11%|██████████████████▋                                                                                                                                                      | 123/1114 [00:01<00:15, 63.30batches/s]
    Epoch  1/1:  13%|█████████████████████▌                                                                                                                                                   | 142/1114 [00:02<00:13, 69.50batches/s]
    Epoch  1/1:  14%|███████████████████████▊                                                                                                                                                 | 157/1114 [00:02<00:15, 61.75batches/s]
    Epoch  1/1:  16%|██████████████████████████▊                                                                                                                                              | 177/1114 [00:02<00:14, 66.86batches/s]
    Epoch  1/1:  18%|█████████████████████████████▌                                                                                                                                           | 195/1114 [00:02<00:12, 70.93batches/s]
    Epoch  1/1:  19%|███████████████████████████████▊                                                                                                                                         | 210/1114 [00:03<00:14, 64.38batches/s]
    Epoch  1/1:  21%|██████████████████████████████████▉                                                                                                                                      | 230/1114 [00:03<00:12, 68.10batches/s]
    Epoch  1/1:  22%|█████████████████████████████████████▌                                                                                                                                   | 248/1114 [00:03<00:12, 71.09batches/s]
    Epoch  1/1:  24%|███████████████████████████████████████▋                                                                                                                                 | 262/1114 [00:03<00:12, 65.79batches/s]
    Epoch  1/1:  25%|███████████████████████████████████████████                                                                                                                              | 284/1114 [00:04<00:12, 69.08batches/s]
    Epoch  1/1:  27%|█████████████████████████████████████████████▋                                                                                                                           | 301/1114 [00:04<00:12, 65.12batches/s]
    Epoch  1/1:  29%|████████████████████████████████████████████████▋                                                                                                                        | 321/1114 [00:04<00:11, 67.86batches/s]
    Epoch  1/1:  31%|████████████████████████████████████████████████████▋                                                                                                                    | 347/1114 [00:04<00:10, 71.83batches/s]
    Epoch  1/1:  33%|███████████████████████████████████████████████████████▏                                                                                                                 | 364/1114 [00:05<00:10, 68.24batches/s]
    Epoch  1/1:  35%|██████████████████████████████████████████████████████████▌                                                                                                              | 386/1114 [00:05<00:10, 70.99batches/s]
    Epoch  1/1:  36%|████████████████████████████████████████████████████████████▉                                                                                                            | 402/1114 [00:05<00:10, 67.45batches/s]
    Epoch  1/1:  38%|███████████████████████████████████████████████████████████████▊                                                                                                         | 421/1114 [00:06<00:09, 69.42batches/s]
    Epoch  1/1:  40%|███████████████████████████████████████████████████████████████████▎                                                                                                     | 444/1114 [00:06<00:09, 72.00batches/s]
    Epoch  1/1:  41%|█████████████████████████████████████████████████████████████████████▉                                                                                                   | 461/1114 [00:06<00:09, 69.08batches/s]
    Epoch  1/1:  43%|█████████████████████████████████████████████████████████████████████████▎                                                                                               | 483/1114 [00:06<00:08, 71.22batches/s]
    Epoch  1/1:  45%|████████████████████████████████████████████████████████████████████████████                                                                                             | 501/1114 [00:07<00:08, 68.72batches/s]
    Epoch  1/1:  47%|███████████████████████████████████████████████████████████████████████████████▊                                                                                         | 526/1114 [00:07<00:08, 71.17batches/s]
    Epoch  1/1:  49%|███████████████████████████████████████████████████████████████████████████████████▍                                                                                     | 550/1114 [00:07<00:07, 73.42batches/s]
    Epoch  1/1:  51%|██████████████████████████████████████████████████████████████████████████████████████▏                                                                                  | 568/1114 [00:08<00:07, 70.77batches/s]
    Epoch  1/1:  53%|█████████████████████████████████████████████████████████████████████████████████████████▌                                                                               | 590/1114 [00:08<00:07, 72.59batches/s]
    Epoch  1/1:  54%|████████████████████████████████████████████████████████████████████████████████████████████                                                                             | 607/1114 [00:08<00:07, 70.25batches/s]
    Epoch  1/1:  57%|███████████████████████████████████████████████████████████████████████████████████████████████▌                                                                         | 630/1114 [00:08<00:06, 71.99batches/s]
    Epoch  1/1:  58%|██████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                      | 651/1114 [00:09<00:06, 70.25batches/s]
    Epoch  1/1:  61%|██████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                  | 675/1114 [00:09<00:06, 72.03batches/s]
    Epoch  1/1:  62%|█████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                               | 696/1114 [00:09<00:05, 73.49batches/s]
    Epoch  1/1:  64%|████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                            | 713/1114 [00:09<00:05, 71.42batches/s]
    Epoch  1/1:  66%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                         | 738/1114 [00:10<00:05, 73.17batches/s]
    Epoch  1/1:  68%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                      | 755/1114 [00:10<00:05, 71.29batches/s]
    Epoch  1/1:  70%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                   | 778/1114 [00:10<00:04, 72.78batches/s]
    Epoch  1/1:  72%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                               | 801/1114 [00:11<00:04, 71.38batches/s]
    Epoch  1/1:  74%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                            | 824/1114 [00:11<00:03, 72.78batches/s]
    Epoch  1/1:  76%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                        | 847/1114 [00:11<00:03, 74.12batches/s]
    Epoch  1/1:  78%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                     | 865/1114 [00:11<00:03, 72.56batches/s]
    Epoch  1/1:  79%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                  | 885/1114 [00:12<00:03, 73.58batches/s]
    Epoch  1/1:  81%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                | 901/1114 [00:12<00:02, 71.88batches/s]
    Epoch  1/1:  83%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                             | 921/1114 [00:12<00:02, 72.89batches/s]
    Epoch  1/1:  84%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                          | 940/1114 [00:12<00:02, 73.79batches/s]
    Epoch  1/1:  86%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                        | 956/1114 [00:13<00:02, 72.13batches/s]
    Epoch  1/1:  88%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                    | 979/1114 [00:13<00:01, 73.29batches/s]
    Epoch  1/1:  90%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                 | 1000/1114 [00:13<00:01, 74.28batches/s]
    Epoch  1/1:  91%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎              | 1017/1114 [00:14<00:01, 72.55batches/s]
    Epoch  1/1:  93%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉           | 1041/1114 [00:14<00:00, 73.70batches/s]
    Epoch  1/1:  95%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍        | 1057/1114 [00:14<00:00, 72.25batches/s]
    Epoch  1/1:  97%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍    | 1084/1114 [00:14<00:00, 73.53batches/s]
    Epoch  1/1:  99%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████  | 1101/1114 [00:15<00:00, 72.18batches/s]
    Epoch  1/1: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:15<00:00, 72.77batches/s]


![png](output_18_2.png)


    Validation accuracy at 0.09986666589975357
    epochs =1, learning_rate =0.5
    

    
    Epoch  1/1:   0%|                                                                                                                                                                                   | 0/1114 [00:00<?, ?batches/s]
    Epoch  1/1:   0%|▏                                                                                                                                                                          | 1/1114 [00:00<09:02,  2.05batches/s]
    Epoch  1/1:   2%|███▉                                                                                                                                                                      | 26/1114 [00:00<00:24, 44.20batches/s]
    Epoch  1/1:   4%|███████▍                                                                                                                                                                  | 49/1114 [00:00<00:15, 70.46batches/s]
    Epoch  1/1:   5%|█████████▎                                                                                                                                                                | 61/1114 [00:01<00:20, 50.98batches/s]
    Epoch  1/1:   7%|████████████▏                                                                                                                                                             | 80/1114 [00:01<00:16, 61.67batches/s]
    Epoch  1/1:   9%|██████████████▋                                                                                                                                                           | 96/1114 [00:01<00:14, 68.70batches/s]
    Epoch  1/1:  10%|████████████████▋                                                                                                                                                        | 110/1114 [00:01<00:17, 57.14batches/s]
    Epoch  1/1:  12%|███████████████████▌                                                                                                                                                     | 129/1114 [00:02<00:15, 63.53batches/s]
    Epoch  1/1:  14%|██████████████████████▉                                                                                                                                                  | 151/1114 [00:02<00:16, 59.18batches/s]
    Epoch  1/1:  16%|██████████████████████████▍                                                                                                                                              | 174/1114 [00:02<00:14, 65.22batches/s]
    Epoch  1/1:  18%|██████████████████████████████▍                                                                                                                                          | 201/1114 [00:03<00:14, 62.93batches/s]
    Epoch  1/1:  20%|██████████████████████████████████▎                                                                                                                                      | 226/1114 [00:03<00:12, 68.47batches/s]
    Epoch  1/1:  22%|█████████████████████████████████████▍                                                                                                                                   | 247/1114 [00:03<00:11, 72.26batches/s]
    Epoch  1/1:  24%|███████████████████████████████████████▉                                                                                                                                 | 263/1114 [00:03<00:12, 66.86batches/s]
    Epoch  1/1:  26%|███████████████████████████████████████████▌                                                                                                                             | 287/1114 [00:04<00:11, 71.13batches/s]
    Epoch  1/1:  27%|█████████████████████████████████████████████▉                                                                                                                           | 303/1114 [00:04<00:12, 66.75batches/s]
    Epoch  1/1:  29%|█████████████████████████████████████████████████▌                                                                                                                       | 327/1114 [00:04<00:11, 70.43batches/s]
    Epoch  1/1:  32%|█████████████████████████████████████████████████████▏                                                                                                                   | 351/1114 [00:05<00:11, 67.82batches/s]
    Epoch  1/1:  34%|█████████████████████████████████████████████████████████▎                                                                                                               | 378/1114 [00:05<00:10, 71.62batches/s]
    Epoch  1/1:  36%|████████████████████████████████████████████████████████████▊                                                                                                            | 401/1114 [00:05<00:10, 68.90batches/s]
    Epoch  1/1:  38%|████████████████████████████████████████████████████████████████                                                                                                         | 422/1114 [00:05<00:09, 71.26batches/s]
    Epoch  1/1:  40%|████████████████████████████████████████████████████████████████████                                                                                                     | 449/1114 [00:06<00:08, 74.51batches/s]
    Epoch  1/1:  42%|██████████████████████████████████████████████████████████████████████▉                                                                                                  | 468/1114 [00:06<00:09, 71.41batches/s]
    Epoch  1/1:  44%|██████████████████████████████████████████████████████████████████████████▉                                                                                              | 494/1114 [00:06<00:08, 74.24batches/s]
    Epoch  1/1:  46%|█████████████████████████████████████████████████████████████████████████████▋                                                                                           | 512/1114 [00:07<00:08, 71.48batches/s]
    Epoch  1/1:  48%|█████████████████████████████████████████████████████████████████████████████████▍                                                                                       | 537/1114 [00:07<00:07, 73.86batches/s]
    Epoch  1/1:  50%|████████████████████████████████████████████████████████████████████████████████████                                                                                     | 554/1114 [00:07<00:07, 71.15batches/s]
    Epoch  1/1:  52%|███████████████████████████████████████████████████████████████████████████████████████▌                                                                                 | 577/1114 [00:07<00:07, 73.12batches/s]
    Epoch  1/1:  54%|███████████████████████████████████████████████████████████████████████████████████████████▏                                                                             | 601/1114 [00:08<00:07, 71.31batches/s]
    Epoch  1/1:  57%|███████████████████████████████████████████████████████████████████████████████████████████████▉                                                                         | 632/1114 [00:08<00:06, 73.87batches/s]
    Epoch  1/1:  58%|██████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                      | 651/1114 [00:09<00:06, 71.65batches/s]
    Epoch  1/1:  61%|██████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                  | 678/1114 [00:09<00:05, 73.77batches/s]
    Epoch  1/1:  63%|██████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                              | 701/1114 [00:09<00:05, 71.99batches/s]
    Epoch  1/1:  65%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                          | 727/1114 [00:09<00:05, 73.90batches/s]
    Epoch  1/1:  67%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                       | 751/1114 [00:10<00:05, 71.94batches/s]
    Epoch  1/1:  70%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                   | 775/1114 [00:10<00:04, 73.51batches/s]
    Epoch  1/1:  72%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                               | 801/1114 [00:11<00:04, 72.35batches/s]
    Epoch  1/1:  75%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                           | 830/1114 [00:11<00:03, 74.25batches/s]
    Epoch  1/1:  76%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                        | 851/1114 [00:11<00:03, 72.62batches/s]
    Epoch  1/1:  78%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                    | 874/1114 [00:11<00:03, 73.97batches/s]
    Epoch  1/1:  81%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                | 901/1114 [00:12<00:02, 72.73batches/s]
    Epoch  1/1:  82%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                             | 918/1114 [00:12<00:02, 73.49batches/s]
    Epoch  1/1:  84%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                          | 938/1114 [00:12<00:02, 74.47batches/s]
    Epoch  1/1:  86%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                        | 954/1114 [00:13<00:02, 72.85batches/s]
    Epoch  1/1:  88%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                     | 975/1114 [00:13<00:01, 73.88batches/s]
    Epoch  1/1:  90%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                 | 1000/1114 [00:13<00:01, 75.13batches/s]
    Epoch  1/1:  91%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎              | 1017/1114 [00:13<00:01, 73.58batches/s]
    Epoch  1/1:  93%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉           | 1041/1114 [00:13<00:00, 74.76batches/s]
    Epoch  1/1:  95%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍        | 1057/1114 [00:14<00:00, 73.26batches/s]
    Epoch  1/1:  97%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊     | 1080/1114 [00:14<00:00, 74.33batches/s]
    Epoch  1/1:  99%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████  | 1101/1114 [00:15<00:00, 73.20batches/s]
    Epoch  1/1: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:15<00:00, 73.76batches/s]
    


![png](output_18_5.png)


    Validation accuracy at 0.09986666589975357
    epochs =1, learning_rate =0.1
    

    Epoch  1/1: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:16<00:00, 69.41batches/s]
    


![png](output_18_8.png)


    Validation accuracy at 0.7761333584785461
    epochs =1, learning_rate =0.05
    

    Epoch  1/1: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:16<00:00, 66.97batches/s]
    


![png](output_18_11.png)


    Validation accuracy at 0.8055999875068665
    epochs =1, learning_rate =0.01
    

    Epoch  1/1: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:15<00:00, 70.53batches/s]
    


![png](output_18_14.png)


    Validation accuracy at 0.8149333596229553
    epochs =1, learning_rate =0.2
    

    Epoch  1/1: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:14<00:00, 74.76batches/s]
    


![png](output_18_17.png)


    Validation accuracy at 0.6770666837692261
    epochs =2, learning_rate =0.2
    

    Epoch  1/2: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:14<00:00, 76.16batches/s]
    Epoch  2/2: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:15<00:00, 72.98batches/s]
    


![png](output_18_20.png)


    Validation accuracy at 0.6836000084877014
    epochs =3, learning_rate =0.2
    

    Epoch  1/3: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:15<00:00, 70.80batches/s]
    Epoch  2/3: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:15<00:00, 72.66batches/s]
    Epoch  3/3: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:15<00:00, 71.13batches/s]
    


![png](output_18_23.png)


    Validation accuracy at 0.6880000233650208
    epochs =4, learning_rate =0.2
    

    Epoch  1/4: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:15<00:00, 72.89batches/s]
    Epoch  2/4: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:14<00:00, 75.04batches/s]
    Epoch  3/4: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:15<00:00, 74.06batches/s]
    Epoch  4/4: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:16<00:00, 69.20batches/s]
    


![png](output_18_26.png)


    Validation accuracy at 0.6897333264350891
    epochs =5, learning_rate =0.2
    

    Epoch  1/5: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:15<00:00, 74.01batches/s]
    Epoch  2/5: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:15<00:00, 73.13batches/s]
    Epoch  3/5: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:15<00:00, 73.14batches/s]
    Epoch  4/5: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:15<00:00, 72.31batches/s]
    Epoch  5/5: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:14<00:00, 75.19batches/s]
    


![png](output_18_29.png)


    Validation accuracy at 0.6940000057220459
    


```python
testing_config(batch_size, epochs=5, learning_rate=0.003)
```

    Epoch  1/5: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:15<00:00, 71.76batches/s]
    Epoch  2/5: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:15<00:00, 73.92batches/s]
    Epoch  3/5: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:14<00:00, 74.89batches/s]
    Epoch  4/5: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:14<00:00, 75.13batches/s]
    Epoch  5/5: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:14<00:00, 75.44batches/s]
    


![png](output_19_1.png)


    Validation accuracy at 0.8191999793052673
    


```python
epochs = 5
learning_rate = 0.003
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
```

## Test
You're going to test your model against your hold out dataset/testing data.  This will give you a good indicator of how well the model will do in the real world.  You should have a test accuracy of at least 80%.


```python
### DON'T MODIFY ANYTHING BELOW ###
# The accuracy measured against the test set
test_accuracy = 0.0

with tf.Session() as session:
    
    session.run(init)
    batch_count = int(math.ceil(len(train_features)/batch_size))

    for epoch_i in range(epochs):
        
        # Progress bar
        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')
        
        # The training cycle
        for batch_i in batches_pbar:
            # Get a batch of training features and labels
            batch_start = batch_i*batch_size
            batch_features = train_features[batch_start:batch_start + batch_size]
            batch_labels = train_labels[batch_start:batch_start + batch_size]

            # Run optimizer
            _ = session.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})

        # Check accuracy against Test data
        test_accuracy = session.run(accuracy, feed_dict=test_feed_dict)


assert test_accuracy >= 0.80, 'Test accuracy at {}, should be equal to or greater than 0.80'.format(test_accuracy)
print('Nice Job! Test Accuracy is {}'.format(test_accuracy))
```

    
    Epoch  1/5:   0%|                                                                                                                                                                                   | 0/1114 [00:00<?, ?batches/s]
    Epoch  1/5:   2%|███▏                                                                                                                                                                     | 21/1114 [00:00<00:06, 179.23batches/s]
    Epoch  1/5:   4%|█████▉                                                                                                                                                                   | 39/1114 [00:00<00:06, 178.34batches/s]
    Epoch  1/5:   6%|██████████                                                                                                                                                               | 66/1114 [00:00<00:05, 198.93batches/s]
    Epoch  1/5:   8%|█████████████▎                                                                                                                                                           | 88/1114 [00:00<00:05, 197.81batches/s]
    Epoch  1/5:  10%|████████████████▌                                                                                                                                                       | 110/1114 [00:00<00:05, 200.67batches/s]
    Epoch  1/5:  12%|████████████████████▎                                                                                                                                                   | 135/1114 [00:00<00:04, 203.60batches/s]
    Epoch  1/5:  14%|███████████████████████▏                                                                                                                                                | 154/1114 [00:00<00:04, 200.89batches/s]
    Epoch  1/5:  16%|██████████████████████████▋                                                                                                                                             | 177/1114 [00:00<00:04, 200.34batches/s]
    Epoch  1/5:  18%|██████████████████████████████▌                                                                                                                                         | 203/1114 [00:00<00:04, 206.12batches/s]
    Epoch  1/5:  20%|██████████████████████████████████▏                                                                                                                                     | 227/1114 [00:01<00:04, 209.25batches/s]
    Epoch  1/5:  23%|█████████████████████████████████████▊                                                                                                                                  | 251/1114 [00:01<00:04, 209.68batches/s]
    Epoch  1/5:  25%|█████████████████████████████████████████▏                                                                                                                              | 273/1114 [00:01<00:04, 203.84batches/s]
    Epoch  1/5:  27%|████████████████████████████████████████████▊                                                                                                                           | 297/1114 [00:01<00:03, 205.78batches/s]
    Epoch  1/5:  29%|█████████████████████████████████████████████████▏                                                                                                                      | 326/1114 [00:01<00:03, 211.21batches/s]
    Epoch  1/5:  32%|████████████████████████████████████████████████████▉                                                                                                                   | 351/1114 [00:01<00:03, 212.60batches/s]
    Epoch  1/5:  34%|████████████████████████████████████████████████████████▊                                                                                                               | 377/1114 [00:01<00:03, 213.34batches/s]
    Epoch  1/5:  36%|████████████████████████████████████████████████████████████▎                                                                                                           | 400/1114 [00:01<00:03, 209.86batches/s]
    Epoch  1/5:  38%|████████████████████████████████████████████████████████████████▍                                                                                                       | 427/1114 [00:02<00:03, 212.50batches/s]
    Epoch  1/5:  40%|███████████████████████████████████████████████████████████████████▊                                                                                                    | 450/1114 [00:02<00:03, 213.01batches/s]
    Epoch  1/5:  42%|███████████████████████████████████████████████████████████████████████▎                                                                                                | 473/1114 [00:02<00:03, 213.56batches/s]
    Epoch  1/5:  45%|███████████████████████████████████████████████████████████████████████████▍                                                                                            | 500/1114 [00:02<00:02, 215.78batches/s]
    Epoch  1/5:  47%|███████████████████████████████████████████████████████████████████████████████                                                                                         | 524/1114 [00:02<00:02, 216.76batches/s]
    Epoch  1/5:  49%|██████████████████████████████████████████████████████████████████████████████████▋                                                                                     | 548/1114 [00:02<00:02, 217.14batches/s]
    Epoch  1/5:  52%|██████████████████████████████████████████████████████████████████████████████████████▋                                                                                 | 575/1114 [00:02<00:02, 219.05batches/s]
    Epoch  1/5:  54%|███████████████████████████████████████████████████████████████████████████████████████████                                                                             | 604/1114 [00:02<00:02, 221.47batches/s]
    Epoch  1/5:  57%|███████████████████████████████████████████████████████████████████████████████████████████████▌                                                                        | 634/1114 [00:02<00:02, 223.62batches/s]
    Epoch  1/5:  59%|███████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                    | 661/1114 [00:02<00:02, 223.55batches/s]
    Epoch  1/5:  62%|███████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                | 687/1114 [00:03<00:01, 222.90batches/s]
    Epoch  1/5:  64%|███████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                            | 713/1114 [00:03<00:01, 223.69batches/s]
    Epoch  1/5:  66%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                        | 738/1114 [00:03<00:01, 224.06batches/s]
    Epoch  1/5:  69%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                    | 766/1114 [00:03<00:01, 224.99batches/s]
    Epoch  1/5:  71%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                | 792/1114 [00:03<00:01, 225.77batches/s]
    Epoch  1/5:  73%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                            | 818/1114 [00:03<00:01, 226.62batches/s]
    Epoch  1/5:  76%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                        | 843/1114 [00:03<00:01, 226.81batches/s]
    Epoch  1/5:  78%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                     | 868/1114 [00:03<00:01, 226.45batches/s]
    Epoch  1/5:  80%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                 | 892/1114 [00:03<00:00, 226.43batches/s]
    Epoch  1/5:  83%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                             | 920/1114 [00:04<00:00, 227.57batches/s]
    Epoch  1/5:  85%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                         | 947/1114 [00:04<00:00, 228.42batches/s]
    Epoch  1/5:  87%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                     | 973/1114 [00:04<00:00, 228.84batches/s]
    Epoch  1/5:  90%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                 | 998/1114 [00:04<00:00, 228.05batches/s]
    Epoch  1/5:  92%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏             | 1022/1114 [00:04<00:00, 227.79batches/s]
    Epoch  1/5:  94%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋          | 1045/1114 [00:04<00:00, 227.23batches/s]
    Epoch  1/5:  96%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████       | 1068/1114 [00:04<00:00, 227.23batches/s]
    Epoch  1/5:  98%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌   | 1091/1114 [00:04<00:00, 227.08batches/s]
    Epoch  1/5: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:04<00:00, 227.26batches/s]
    Epoch  2/5:   0%|                                                                                                                                                                                   | 0/1114 [00:00<?, ?batches/s]
    Epoch  2/5:   2%|███▊                                                                                                                                                                     | 25/1114 [00:00<00:04, 242.11batches/s]
    Epoch  2/5:   4%|███████▍                                                                                                                                                                 | 49/1114 [00:00<00:04, 238.41batches/s]
    Epoch  2/5:   6%|██████████▉                                                                                                                                                              | 72/1114 [00:00<00:04, 233.92batches/s]
    Epoch  2/5:   8%|█████████████▊                                                                                                                                                           | 91/1114 [00:00<00:04, 215.59batches/s]
    Epoch  2/5:  10%|████████████████▋                                                                                                                                                       | 111/1114 [00:00<00:04, 212.66batches/s]
    Epoch  2/5:  13%|█████████████████████                                                                                                                                                   | 140/1114 [00:00<00:04, 224.28batches/s]
    Epoch  2/5:  15%|█████████████████████████▊                                                                                                                                              | 171/1114 [00:00<00:04, 234.67batches/s]
    Epoch  2/5:  18%|█████████████████████████████▌                                                                                                                                          | 196/1114 [00:00<00:03, 234.17batches/s]
    Epoch  2/5:  20%|█████████████████████████████████▊                                                                                                                                      | 224/1114 [00:00<00:03, 240.02batches/s]
    Epoch  2/5:  22%|█████████████████████████████████████▌                                                                                                                                  | 249/1114 [00:01<00:03, 240.46batches/s]
    Epoch  2/5:  25%|█████████████████████████████████████████▊                                                                                                                              | 277/1114 [00:01<00:03, 241.94batches/s]
    Epoch  2/5: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:04<00:00, 252.38batches/s]
    Epoch  3/5: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:04<00:00, 255.58batches/s]
    Epoch  4/5: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:04<00:00, 253.40batches/s]
    Epoch  5/5: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1114/1114 [00:04<00:00, 250.10batches/s]
    

    Nice Job! Test Accuracy is 0.878600001335144
    

# Multiple layers
Good job!  You built a one layer TensorFlow network!  However, you might want to build more than one layer.  This is deep learning after all!  In the next section, you will start to satisfy your need for more layers.
