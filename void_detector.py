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

def conv_layer(in_tensor,name, kernel_size, n_output_channels, strides=(1,1,1,1), padding_mode='SAME', act=True):
    
    with tf.variable_scope(name):
        input_shape = in_tensor.get_shape().as_list()
        n_input_channels = input_shape[-1]
        weights_shape = list(kernel_size)+[n_input_channels, n_output_channels]
        
        weights = tf.get_variable(name='_weights', shape=weights_shape, initializer=tf.random_normal_initializer()) #mean=0.5, stddev=0.02
        conv = tf.nn.conv2d(in_tensor, weights, strides=strides, padding=padding_mode)
        
        mean, variance = tf.nn.moments(conv, axes=[0, 1, 2])
        conv = tf.nn.batch_normalization(conv, mean=mean, variance=variance, offset=None, scale=None, variance_epsilon=0.001)
        if act:
            conv = tf.nn.relu(conv, name='activation')
        tensor_info(weights, conv)
        return conv
        


     
def build_graph(learning_rate, n_i=1):        
        
    print('..............INPUT.................')
    in_tensor = tf.placeholder(dtype=tf.float32, shape=(n_i,120,160,3), name='input')
    
    tf_y = tf.placeholder(tf.float32, shape=[n_i,120,160,1], name='tf_y')  
#    mean, variance = tf.nn.moments(tf_y, axes=[0, 1, 2])
#    tf_y = tf.nn.batch_normalization(tf_y, mean=mean, variance=variance, offset=None, scale=None, variance_epsilon=0.001)    
#    tf_y = tf.nn.relu(tf_y, name='tf_y_relu')
    tensor_info(in_tensor, tf_y) 


    ks = (2,2) # KERNEL-size for ENcoding layers
    print('..............CONV LAYER ENCODER 1...............')
#    c0 = conv_layer(in_tensor, 'conv_0', (2,2), 16)
    c1 = conv_layer(in_tensor, 'conv_1', ks, 16)
    p1 = tf.nn.max_pool(c1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    tensor_info(p1)
    print('..............CONV LAYER 2...............')
    c2 = conv_layer(p1, 'conv_2', ks, 32)
    c3 = conv_layer(c2, 'conv_3', ks, 32)
    p2 = tf.nn.max_pool(c3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')    
    tensor_info(p2)
    print('..............CONV LAYER 3...............')
    c4 = conv_layer(p2, 'conv_4', ks, 64)
    c5 = conv_layer(c4, 'conv_5', ks, 64)
    p3 = tf.nn.max_pool(c5, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    tensor_info(p3)
    
    #============================= DECODING ======================================
    
    ks_decoder = (5,5) # KERNEL-size for DEcoding layers
    print('..............CONV LAYER DECODER 1...............')
    output1 = tf.stack([n_i, 30,40,64]) 
    w1 = tf.get_variable('w1', shape=(2,2,64,64), initializer=tf.random_normal_initializer())
    upsample_1 = tf.nn.conv2d_transpose(p3, w1, output_shape=output1, strides=[1,2,2,1], padding='VALID', name='upsample_1')
    tensor_info(w1, upsample_1)
    c6 = conv_layer(upsample_1, 'conv_6', ks_decoder, 64, act=False)
    c7 = conv_layer(c6, 'conv_7', ks_decoder, 64, act=False)
        
    print('..............CONV LAYER DECODER 2...............')
    output2 = tf.stack([n_i, 60,80,32]) 
    w2 = tf.get_variable('w2', shape=(1,1,32,64), initializer=tf.random_normal_initializer())
    upsample_2 = tf.nn.conv2d_transpose(c7, w2, output_shape=output2, strides=[1,2,2,1], padding='VALID', name='upsample_2')
    tensor_info(w2, upsample_2)
    c8 = conv_layer(upsample_2, 'conv_8', ks_decoder, 32, act=False)
    c9 = conv_layer(c8, 'conv_9', ks_decoder, 32, act=False)
    
    print('..............CONV LAYER DECODER 3...............')
    output3 = tf.stack([n_i, 120,160,16]) 
    w3 = tf.get_variable('w3', shape=(1,1,16,32), initializer=tf.random_normal_initializer())
    upsample_3 = tf.nn.conv2d_transpose(c9, w3, output_shape=output3, strides=[1,2,2,1], padding='VALID', name='upsample_3')
    tensor_info(w3, upsample_3)
    c10 = conv_layer(upsample_3, 'conv_10', ks_decoder, 16, act=False)
    c11 = conv_layer(c10, 'conv_11', ks_decoder, 1, act=False)
    
    
    #=================================== RESULTS ================================================================
    print('..............RESULTS...............')   
    cross = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_y, logits=c11, name='softmax')
    cross_entropy_loss = tf.reduce_sum(tf.sqrt(tf.square(cross)), name='cross_entropy_loss')
    tensor_info(cross, cross_entropy_loss)
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = optimizer.minimize(cross_entropy_loss, name='train_op')   
    
    
    #============================ Mitschreiben =======================================   
    logdir = '../{}/run-{}/'.format('logs', datetime.utcnow().strftime('%m%d%H%M'))
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())       
    return file_writer


# ================================== Funktionen zum Kontrolieren ============================    
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
    saver.save(sess, os.path.join(path+model_name, model_name), global_step=epoch)    
    print('Modell speichern in {}'.format(path))
    
def predict(X, sess, model_name='sigmoid_model'):
    
    feed = {'input:0': X, 'tf_y:0': np.zeros(shape=(*np.shape(X)[:3],1))}
    prediction = sess.run(['conv_11/activation:0'], feed_dic=feed)
    return prediction
#=============================================================================================
        
        

if __name__=='__main__':
    
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        # BUILD GRAPH
        build_graph(learning_rate=0.00001, n_i=12)
        
        p = """../model/cnn_model/cnn_model-0""" # for images (600,800)
        p_sig = """../model/sigmoid_model/sigmoid_model-0"""
        p_sig_02 = """../model/sigmoid_model_02/sigmoid_model_02-0"""
        p_sig_04 = """../model/sigmoid_model_04/sigmoid_model_04-0"""
        p_sig_05 = """../model/sigmoid_model_05/sigmoid_model_05-0"""
        p_sig_06 = """../model/sigmoid_model_06/sigmoid_model_06-0"""  # for images (100,160)
        p_sig_06_final = """../model/sigmoid_model_06_final/sigmoid_model_06_final-61"""  # for images (120,160)
        p_sig_07 = """../model/sigmoid_model_07/sigmoid_model_07-100"""
        p_sig_08 = """../model/sigmoid_model_08/sigmoid_model_08-0"""
        saver = init_variables(sess, path=p_sig_06_final)
        
        
        start = time.time()
        for n in range(1,2):
            avg_loss = 0.0
            gen = load_input(bs=12)
            
            for (x,y),_ in zip(gen, range(2)):
                feed = {'input:0': x, 'tf_y:0': y}
                loss, tf_y, c11, softmax = sess.run(['cross_entropy_loss:0', 'tf_y:0', 'conv_11/batchnorm/add_1:0', 'softmax:0'], feed_dict=feed)

                avg_loss += loss
            print('{} | Avg_Loss: {} | Time: {}'.format(n, avg_loss, (time.time()-start)/60))
            
            if n % 5 == 0:
                save(saver, sess, 0, model_name='sigmoid_model_08') # 
        
        
        fig = plt.figure()
        m = 1
        y = 12
        for i in range(0,y+0):
            plt.subplot(y,3,m)
            plt.imshow(tf_y[i,:,:,0])
            m +=1
        
            plt.subplot(y,3,m)
            plt.imshow(c11[i,:,:,0])
            m +=1
            
            plt.subplot(y,3,m)
            plt.imshow(softmax[i,:,:,0])
            m+=1
        
        plt.show()
        print('softmax: ', softmax.max(), softmax.min())     
        print('c11: ',c11.max(), c11.min()) 
        print('tf_y_relu: ', tf_y.max(), tf_y.min())
        
#%%     
    plt.figure()
    plt.subplot(1,3,1)
    mask = cv2.resize(c11[8], (1600,1200))
    ret, mask = cv2.threshold(mask,1.2,255,cv2.THRESH_BINARY)
    plt.imshow(mask)

    plt.subplot(1,3,2)
    img = cv2.imread('../Mikro_unmarkiert/A_62_mikro_1.jpg')
    plt.imshow(img)
    

    plt.subplot(1,3,3)
    comb = img
    comb[:,:,0] = np.array(mask, dtype=np.uint8)
    plt.imshow(comb)
    cv2.imshow('comb',comb)
    cv2.imwrite('comb.jpg', comb)
        
        
        
        
        
        
        
        