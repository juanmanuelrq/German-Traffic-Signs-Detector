import click
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import tensorflow as tf
import cv2
import os
import shutil
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

IMG_SIZE = 32 

from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    layer_depth = {
        'layer_1' : 6,
        'layer_2' : 16,
        'layer_3' : 120,
        'layer_f1' : 84
    }

    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_w = tf.Variable(tf.truncated_normal(shape = [5,5,1,6],mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x,conv1_w, strides = [1,1,1,1], padding = 'VALID') + conv1_b 
    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    pool_1 = tf.nn.max_pool(conv1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b
    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID') 
    
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten(pool_2)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape = (400,120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1,fc1_w) + fc1_b
    
    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1,fc2_w) + fc2_b
    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)
    
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_w = tf.Variable(tf.truncated_normal(shape = (84,10), mean = mu , stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    return logits


@click.group()
@click.option('--m', default=1, help='Model')
def model(m):
    pass



@model.command()
def download():
    click.echo('Initialized the download')
    import urllib2
    #url = 'https://www.dynaexamples.com/examples-manual/ls-dyna_example.zip'
    url = 'http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip'
    filename = url.split("?")[0].split("/")[-1]
    #download zip file
    response = urllib2.urlopen(url) 
    zipcontent= response.read()    
    with open(filename, 'w') as f:
    	f.write(zipcontent)
    click.echo('finished the download')

    import zipfile
    zip_ref = zipfile.ZipFile(filename, 'r')
    

    zip_ref.extractall('images')
    zip_ref.close()


@model.command()
def create_train_test():
    path='images'    
    image_files = []
    labels = []
    for x in enumerate(["%.2d" % i for i in range(43)]):

        for image_file in os.listdir('images/FullIJCNN2013/'+str(x[1])):
            image_files.append('images/FullIJCNN2013/' + str(x[1]) +'/'+ image_file)
            labels.append(str(x[1]))

    images = []
    for image_file in image_files:
        #img = cv2.imread(image_file)             
        images.append(image_file) 

    trainX, x_test, trainY, y_test = train_test_split(images, labels, test_size=0.2)  

    shutil.rmtree('images/train')
    shutil.rmtree('images/test')
    
    for image in enumerate(trainX):
        img = cv2.imread(image[1]) 
        name_file = str(image[1].replace('images/FullIJCNN2013', 'images/train'))
        file_path = name_file
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        cv2.imwrite(name_file,img)
    

    for image in enumerate(x_test):
        img = cv2.imread(image[1]) 
        name_file = image[1].replace('images/FullIJCNN2013', 'images/test') 
        file_path = name_file
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)  
        cv2.imwrite(name_file,img)


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)



def load_train_test():
    test_directory = 'images/test'

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for root, dirs, files in os.walk(test_directory):        
        for name in files:
            if name.endswith((".ppm")):
                img = cv2.imread(root + '/' +name)
                img = cv2.resize(img,(28,28),interpolation = cv2.INTER_CUBIC)
                x_test.append(img)
                y_test.append(int(root.replace(test_directory+'/','')))

    train_directory = 'images/train'

    for root, dirs, files in os.walk(train_directory):        
        for name in files:
            if name.endswith((".ppm")):
                img = cv2.imread(root + '/' +name)
                img = cv2.resize(img,(28,28),interpolation = cv2.INTER_CUBIC)
                x_train.append(img)
                y_train.append(int(root.replace(train_directory+'/','')))

    return x_train,y_train,x_test,y_test



@model.command()
@click.option('--m', default=1, help='select model: 1(sklearn),2(softmax-tensorflow),3(lenet)')
def train(m):
    if m == 1:
        path='images'    
        image_files = []
        labels = []
        for x in enumerate(["%.2d" % i for i in range(43)]):

            for image_file in os.listdir('images/FullIJCNN2013/'+str(x[1])):
                image_files.append('images/FullIJCNN2013/' + str(x[1]) +'/'+ image_file)
                labels.append(str(x[1]))


        images = []
        for image_file in image_files:
            image = Image.open(image_file)
            image = image.convert('RGB')
            image = image.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
            image = np.array(list(image.getdata()), dtype='uint8')
            image = np.reshape(image, (32, 32, 3))       
            images.append(image)        

        X = np.reshape(images,(len(image_files),32*32*3))
        y = labels
        
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)   
       
        from sklearn.linear_model import LogisticRegression
        logisticRegr = LogisticRegression()
        logisticRegr.fit(x_train, y_train)
        logisticRegr.predict(x_test[0].reshape(1,-1))
        score = logisticRegr.score(x_test, y_test)
        click.echo('score: ' + str(score))

    if m == 2:
        #reference https://www.tensorflow.org/tutorials/wide
        #reference https://stackoverflow.com/questions/37454932/tensorflow-train-step-feed-incorrect
        x_train,y_train,x_test,y_test = load_train_test()

        values = np.array(y_train)
        n_values = np.max(values) + 1
        y_train = np.eye(n_values)[values]


        values = np.array(y_test)
        n_values = np.max(values) + 1
        y_test = np.eye(n_values)[values]
 

        x_train = np.reshape(x_train,(len(x_train),28*28*3))          
        x_test = np.reshape(x_test,(len(x_test),28*28*3))
      
        x = tf.placeholder(tf.float32, [None, x_train.shape[1]])
        W = tf.Variable(tf.zeros([x_train.shape[1], y_train.shape[1]]))
        b = tf.Variable(tf.zeros([y_train.shape[1]]))
        y = tf.matmul(x, W) + b             
        y_ = tf.placeholder(tf.float32, [None, y_train.shape[1]])

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        for _ in range(1000):
            batch_xs, batch_ys = next_batch(100,x_train,y_train)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))

    if m==3:
        #reference https://github.com/sujaybabruwad/LeNet-in-Tensorflow/blob/master/LeNet-Lab.ipynb
        #reference https://github.com/mohamedameen93/German-Traffic-Sign-Classification-Using-TensorFlow/blob/master/Traffic_Sign_Classifier.ipynb
        click.echo('this model is in under design') 



@model.command()
@click.option('--m', default=1, help='select model: 1(sklearn),2(softmax-tensorflow),3(lenet)')
def test(m):
    print m
    pass


@model.command()
def infer(model):
    pass


if __name__ == '__main__':
    model(obj={})