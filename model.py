import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
from PIL import Image
from load_input_images import load_input
import cv2
import matplotlib.pyplot as plt


def tensor_info(*tensors):
    for tensor in tensors:
        print(tensor.name, tensor.get_shape())

def conv_layer(in_tensor,name, kernel_size, n_output_channels, strides=(1,1,1,1), padding_mode='SAME'):
    
    with tf.variable_scope(name):
        input_shape = in_tensor.get_shape().as_list()
        n_input_channels = input_shape[-1]
        weights_shape = list(kernel_size)+[n_input_channels, n_output_channels]
        
        weights = tf.get_variable(name='_weights', shape=weights_shape, initializer=tf.random_normal_initializer(mean=0.5, stddev=0.02))
        biases = tf.get_variable(name='_biases', initializer=tf.zeros(shape=[n_output_channels]))
        conv = tf.nn.conv2d(in_tensor, weights, strides=strides, padding=padding_mode)
        conv = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(conv, name='activation')
        tensor_info(weights, biases, conv)
        return conv
        
def fc_layer(input_tensor, name, n_output_units, activation_fn=None):
    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()[1:]
        n_input_units = np.prod(input_shape)

        if len(input_shape) > 1:
            input_tensor = tf.reshape(input_tensor, shape=(-1, n_input_units))        
        weights_shape = [n_input_units, n_output_units]
        
        weights = tf.get_variable(name='_weights', shape=weights_shape)
        biases = tf.get_variable(name='_biases', initializer=tf.zeros(shape=[n_output_units]))
        layer = tf.matmul(input_tensor, weights)
        if activation_fn is None:
            tensor_info(weights, biases, layer)
            return layer
        layer = activation_fn(layer, name='activation')
        tensor_info(weights, biases, layer)
        return layer        
        
def build_graph(learning_rate):        
        

    print('..............INPUT.................')
    in_tensor = tf.placeholder(dtype=tf.float32, shape=(4,600,800,3), name='input')
    tf_y = tf.placeholder(tf.float32, shape=[4,600,800,1], name='tf_y')    
    tensor_info(in_tensor, tf_y)
    print('..............CONV LAYER 1...............')
    c1 = conv_layer(in_tensor, 'conv_1', (2,2), 3, strides=(1,1,1,1), padding_mode='VALID')
    p1 = tf.nn.max_pool(c1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    tensor_info(p1)
    print('..............CONV LAYER 2...............')
    c2 = conv_layer(p1, 'conv_2', (2,2), 6, strides=(1,1,1,1), padding_mode='VALID')
    p2 = tf.nn.max_pool(c2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    tensor_info(p2)
    print('..............CONV LAYER 3...............')
    c3 = conv_layer(p2, 'conv_3', (2,2), 12, strides=(1,1,1,1), padding_mode='VALID')
    p3 = tf.nn.max_pool(c3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    tensor_info(p3)
    
    print('..............CONV LAYER 4 UP...............')
    output1 = tf.stack([4, 150,200,9]) 
    w1 = tf.get_variable('w1', shape=(2,2,9,12), initializer=tf.random_normal_initializer(mean=0.5, stddev=0.02))
    p4 = tf.nn.conv2d_transpose(p3, w1, output_shape=output1, strides=[1,2,2,1], padding='SAME')
    tensor_info(w1, p4)

    print('..............CONV LAYER 5 UP...............')
    output2 = tf.stack([4, 300,400,6]) 
    w2 = tf.get_variable('w2', shape=(2,2,6,9), initializer=tf.random_normal_initializer(mean=0.5, stddev=0.02))
    p5 = tf.nn.conv2d_transpose(p4, w2, output_shape=output2, strides=[1,2,2,1], padding='SAME')
    tensor_info(w2, p5) 
    
    print('..............CONV LAYER 6 UP...............')
    output3 = tf.stack([4, 600,800,1]) 
    w3 = tf.get_variable('w3', shape=(2,2,1,6), initializer=tf.random_normal_initializer(mean=0.5, stddev=0.02))
    p6 = tf.nn.conv2d_transpose(p5, w3, output_shape=output3, strides=[1,2,2,1], padding='SAME', name='output')
    tensor_info(w3, p6) 
    
    soft_output = tf.nn.softmax(p6, name='soft_output')
    soft_input = tf.nn.softmax(tf_y, name='soft_input')
    # Verlustfunktion und Optimierung
    
    
    cross_entropy_loss = tf.reduce_mean(tf.square(soft_output-soft_input), name='cross_entropy_loss')
#    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_y, logits=p6), name='cross_entropy_loss')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = optimizer.minimize(cross_entropy_loss, name='train_op')
    
    # Vorhersage und Dokumentation
#    predictions = {'probabilities': tf.nn.softmax(p6, name='probabilities'),
#               'labels': tf.cast(tf.argmax(f2, axis=1), dtype=tf.int32, name='labels')}
#    correct_predictions = tf.equal(predictions['labels'], tf_y, name='correct_preds')
#    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
#    summary_accuracy = tf.summary.scalar('ACCURACY', accuracy)
#    summary_loss = tf.summary.scalar('LOSS', cross_entropy_loss)
    
    # Mitschreiben    
    logdir = '../{}/run-{}/'.format('logs', datetime.utcnow().strftime('%m%d%H%M'))
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())       
    return file_writer

def init_variables(sess, path=None):
    saver = tf.train.Saver()
    if path != None:
        saver.restore(sess, path)      
    else:
        sess.run(tf.global_variables_initializer())
    return saver

def save(saver, sess, epoch, path='../model/', model_name='cnn_model'):
    if not os.path.isdir(path):
        os.makedirs(path+model_name)
    print('Modell speichern in {}'.format(path))
    saver.save(sess, os.path.join(path+model_name, model_name), global_step=epoch)
    
    
def train(sess, epochs, file_writer, saver):
    start = time.time()
    for epoch in range(epochs):
        avg_loss = 0.0
        bg = load_input()
        for tf_input, tf_y in bg:
            feed = {'input:0': tf_input/255, 'tf_y:0': tf_y/255}
            loss,_,soft_output,soft_input = sess.run(['cross_entropy_loss:0', 'train_op', 'output:0', 'input:0'], feed_dict=feed)
            avg_loss += loss
#            print(loss)

        print('{} | Avg_Loss: {} | Time: {}'.format(epoch, avg_loss, (time.time()-start)/60))
#        if validation_set:
#            bg_v = load_input()
#            for (x_val, y_val) , _ in zip(bg_v, range(1)):
#                feed = {'input:0': x_val, 'tf_y:0': y_val}
#                valid_acc = sess.run('accuracy:0', feed_dict=feed)
#                print('{}: Loss: {:3.4f} --- Validierung: {:7.3f} '.format(epoch, avg_loss, valid_acc), end=' ')
#                s_a, s_l = sess.run(['ACCURACY:0','LOSS:0'],feed_dict=feed)
#                file_writer.add_summary(s_a, epoch)
        
        
    save(saver, sess, 0)
    return soft_output,soft_input
    
    
def predict(sess, X):
    feed = {'input:0': X}
    results = sess.run(['output:0'], feed_dict= feed)
        
    return results
    
if __name__=='__main__':
    
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        # BUILD GRAPH
        file_writer = build_graph(learning_rate=1e-4)
        
        # INIT VARIABLES / RESTORE VARIABLES
        p = """../model/cnn_model/cnn_model-0"""
        saver = init_variables(sess, path=None)        
        p6,soft_input = train(sess, 1, file_writer, saver)
        
        img = cv2.imread('../Mikro_unmarkiert/A_62_mikro_5.jpg')
        img = cv2.resize(img, (800,600))
        z = np.zeros((4,600,800,3))
        z[-1] = img
        results = predict(sess, z)
#        cv2.imshow('prediction', results[0][-1])
        plt.imshow(results[0][3,:,:,0])
        plt.figure(2)
        plt.imshow(soft_input[3,:,:,0])
        plt.figure(3)
        plt.imshow(p6[3,:,:,0])        
        
        
        
        

#        
#        # GET_DATA        
#        path = '/home/anton/Schreibtisch/DataScienceTraining/02_kaggle competitions/Humpback Whale Identification Challenge/new_train'#'/home/anton/Schreibtisch/DataScienceTraining/03_self.projects/gestures/train'#'
#        train_num = 72*3
#        
#        # GESTURES
##        X = np.array(os.listdir(path))[:train_num]
##        y = np.array([i[-5] for i in X])[:train_num]
#        
#        # WHALE
#        df = pd.read_csv('../new_train_data.csv')        
#        X = df.Image.values#[:train_num]
#        y = df.digits.values#[:train_num]
#        
#        
#        
#        # TRAIN GRAPH
##        train(sess, X, y, 20,  file_writer,saver, validation_set=True)
#        
#        
#        # PREDICTION
#        df_to_pred = pd.read_csv('../test_image_data.csv')
#        X = df_to_pred.Image.values
#        y = np.zeros((len(X)))U-Net: Convolutioanl Networks for Biomedical Image Segmentation. Ronneberger et al. 2015
#        
#        results = predict(sess, X,y)
#        results = np.array(results).reshape(15610,5)
#        
#        
#        r = pd.DataFrame(results, columns=('5','4','3','2','1'))
#        new_df = pd.concat([df_to_pred, r], axis=1)
##%%
#        j = 0
#        new_df['submission'] = 'none'
#        for img in new_df.Image:
#            sub = []
#            for rang in [*'12345']:
#                value = new_df[new_df.Image==img][rang].values[0]
#                idnt = df[df.digits==value]['Id'].iloc[0]
#                sub.append(idnt)
#                
#            sub = str(sub).replace(',','').strip('[]').replace('\'','').replace('\'','')
#            new_df.loc[new_df.Image==img, 'submission'] = sub
##            if j == 0:
##                break
#        
#        
#        
#        
#        
#        
#        new_df.to_csv('../submission_0426.csv')

        
        
        
        
        
        
        
        
        