# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 2020

@author: Chenlin
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
name = 'weight'

def conv2D(input, filters, kernel_size=3, stride=1, padding='SAME', d_rate=1):
    return tf.layers.conv2d(input, filters=filters, kernel_size=kernel_size,
                            padding=padding, dilation_rate=d_rate, strides=stride,
                            kernel_initializer=tf.variance_scaling_initializer())

def dilated_conv2D_layer(inputs,num_outputs, name= 'weight',kernel_size = 3,activation_fn = None,rate = 2,padding = 'SAME'):

    in_channels = inputs.get_shape().as_list()[3]
    kernel = [kernel_size, kernel_size, in_channels, num_outputs]

    filter_weight = slim.variable(name= name,
                                  shape=kernel,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
    inputs = tf.nn.atrous_conv2d(inputs, filter_weight, rate=rate, padding=padding)
    if not activation_fn is None:
        inputs = activation_fn(inputs)
    return inputs

def bn(input, is_training=True):
    return tf.layers.batch_normalization(inputs=input, training=is_training, center=True, scale=True, fused=True)

def aspp(input):
    p1 = bn(conv2D(input, 256, 1, 1), is_training=True)
    p01 = bn(conv2D(input, 256, 3, 1, d_rate=1), is_training=True)
    p02 = bn(conv2D(input, 256, 3, 1, d_rate=3), is_training=True)
    p03 = bn(conv2D(input, 256, 3, 1, d_rate=6), is_training=True)

    p = tf.concat([p01, p02, p03], axis=-1)
    p = conv2D(p, 256, 1)

    return p

    return asp

def attention(x1, x2, is_training, channel = 256):
    x1=conv2D(x1,channel,1)
    x1=bn(x1,is_training)

    x2=conv2D(x2,channel,1)
    x2=bn(x2,is_training)

    x=tf.nn.relu(x1+x2)
    x=conv2D(x,1,1)
    x=bn(x,is_training)

    att=tf.nn.sigmoid(x)

    return att

def att_unet(input,is_training=True):
    """attition unet"""
    conv1 = conv2D(input, 32)
    bn1 = tf.nn.relu(bn(conv1, is_training))
    conv1_1 = conv2D(bn1, 32)
    bn1_1 = tf.nn.relu(bn(conv1_1, is_training))
    pool1 = tf.layers.max_pooling2d(bn1_1, 2, 2)

    conv2 = conv2D(pool1, 64)
    bn2 = tf.nn.relu(bn(conv2, is_training))
    conv2_1 = conv2D(bn2, 64)
    bn2_1 = tf.nn.relu(bn(conv2_1, is_training))
    pool2 = tf.layers.max_pooling2d(bn2_1, 2, 2)

    conv3 = conv2D(pool2, 128)
    bn3 = tf.nn.relu(bn(conv3, is_training))
    conv3_1 = conv2D(bn3, 128)
    bn3_1 = tf.nn.relu(bn(conv3_1, is_training))
    pool3 = tf.layers.max_pooling2d(bn3_1, 2, 2)

    conv4 = conv2D(pool3, 256)
    bn4 = tf.nn.relu(bn(conv4, is_training))
    conv4_1 = conv2D(bn4, 256)
    bn4_1 = tf.nn.relu(bn(conv4_1, is_training))
    pool4 = tf.layers.max_pooling2d(bn4_1, 2, 2)

    conv5 = conv2D(pool4, 512)
    bn5 = tf.nn.relu(bn(conv5, is_training))
    conv5_1 = conv2D(bn5, 512)
    bn5_1 = tf.nn.relu(bn(conv5_1, is_training))

    up1 = tf.image.resize_images(bn5_1, tf.shape(bn5_1)[1:3] * 2)
    up1 = conv2D(up1, 256)
    up1 = attention(up1, bn4_1,is_training,256)
    up1 = conv2D(up1, 256)
    up1 = tf.nn.relu(bn(up1, is_training))
    up1 = conv2D(up1, 256)
    up1 = tf.nn.relu(bn(up1, is_training))

    up2 = tf.image.resize_images(up1, tf.shape(up1)[1:3] * 2)
    up2 = conv2D(up2, 128)
    up2 = attention(up2, bn3_1, is_training,128)
    up2 = conv2D(up2, 128)
    up2 = tf.nn.relu(bn(up2, is_training))
    up2 = conv2D(up2, 128)
    up2 = tf.nn.relu(bn(up2, is_training))

    up3 = tf.image.resize_images(up2, tf.shape(up2)[1:3] * 2)
    up3 = conv2D(up3, 64)
    up3 = attention(up3, bn2_1, is_training,64)
    up3 = conv2D(up3, 64)
    up3 = tf.nn.relu(bn(up3, is_training))
    up3 = conv2D(up3, 64)
    up3 = tf.nn.relu(bn(up3, is_training))

    up4 = tf.image.resize_images(up3, tf.shape(up3)[1:3] * 2)
    up4 = conv2D(up4, 32)
    up4 = attention(up4, bn1_1,is_training,32)
    up4 = conv2D(up4, 32)
    up4 = tf.nn.relu(bn(up4, is_training))
    up4 = conv2D(up4, 32)
    up4 = tf.nn.relu(bn(up4, is_training))

    logits = conv2D(up4, 1, 1)

    return logits

def dilated_aspp_unet(input,is_training=True):
    """aspp(+dilated) unet"""
    conv1 = dilated_conv2D_layer(input, 32, 'conv1')
    bn1 = tf.nn.relu(bn(conv1, is_training))
    conv1_1 = dilated_conv2D_layer(bn1, 32, 'conv1_1')
    bn1_1 = tf.nn.relu(bn(conv1_1, is_training))
    pool1 = tf.layers.max_pooling2d(bn1_1, 2, 2)

    conv2 = dilated_conv2D_layer(pool1, 64, 'conv2')
    bn2 = tf.nn.relu(bn(conv2, is_training))
    conv2_1 = dilated_conv2D_layer(bn2, 64, 'conv2_1')
    bn2_1 = tf.nn.relu(bn(conv2_1, is_training))
    pool2 = tf.layers.max_pooling2d(bn2_1, 2, 2)

    conv3 = dilated_conv2D_layer(pool2, 128, 'conv3')
    bn3 = tf.nn.relu(bn(conv3, is_training))
    conv3_1 = dilated_conv2D_layer(bn3, 128, 'conv3_1')
    bn3_1 = tf.nn.relu(bn(conv3_1, is_training))
    pool3 = tf.layers.max_pooling2d(bn3_1, 2, 2)

    conv4 = dilated_conv2D_layer(pool3, 256, 'conv4')
    bn4 = tf.nn.relu(bn(conv4, is_training))
    conv4_1 = dilated_conv2D_layer(bn4, 256, 'conv4_1')
    bn4_1 = tf.nn.relu(bn(conv4_1, is_training))

    ap = aspp(bn4_1)

    up2 = tf.image.resize_images(ap, tf.shape(ap)[1:3] * 2)
    up2 = tf.concat([up2, bn3_1], axis=3)
    up2 = dilated_conv2D_layer(up2, 128, 'up2')
    up2 = tf.nn.relu(bn(up2, is_training))
    up2 = dilated_conv2D_layer(up2, 128, 'up2_1')
    up2 = tf.nn.relu(bn(up2, is_training))

    up3 = tf.image.resize_images(up2, tf.shape(up2)[1:3] * 2)
    up3 = dilated_conv2D_layer(up3, 64, 'up3')
    up3 = tf.concat([up3, bn2_1], axis=3)
    up3 = dilated_conv2D_layer(up3, 64, 'up3_1')
    up3 = tf.nn.relu(bn(up3, is_training))
    up3 = dilated_conv2D_layer(up3, 64, 'up3_2')
    up3 = tf.nn.relu(bn(up3, is_training))

    up4 = tf.image.resize_images(up3, tf.shape(up3)[1:3] * 2)
    up4 = dilated_conv2D_layer(up4, 32, 'up4')
    up4 = tf.concat([up4, bn1_1], axis=3)
    up4 = dilated_conv2D_layer(up4, 32, 'up4_1')
    up4 = tf.nn.relu(bn(up4, is_training))
    up4 = dilated_conv2D_layer(up4, 32, 'up4_2')
    up4 = tf.nn.relu(bn(up4, is_training))

    logits = conv2D(up4, 1, 1)

    return logits

def da_u_net(input,is_training=True):
    """attention+aspp(+dilated)+unet"""
    conv1 = dilated_conv2D_layer(input, 32, 'conv1')
    bn1 = tf.nn.relu(bn(conv1, is_training))
    conv1_1 = dilated_conv2D_layer(bn1, 32, 'conv1_1')
    bn1_1 = tf.nn.relu(bn(conv1_1, is_training))
    pool1 = tf.layers.max_pooling2d(bn1_1, 2, 2)

    conv2 = dilated_conv2D_layer(pool1, 64, 'conv2')
    bn2 = tf.nn.relu(bn(conv2, is_training))
    conv2_1 = dilated_conv2D_layer(bn2, 64, 'conv2_1')
    bn2_1 = tf.nn.relu(bn(conv2_1, is_training))
    pool2 = tf.layers.max_pooling2d(bn2_1, 2, 2)

    conv3 = dilated_conv2D_layer(pool2, 128, 'conv3')
    bn3 = tf.nn.relu(bn(conv3, is_training))
    conv3_1 = dilated_conv2D_layer(bn3, 128, 'conv3_1')
    bn3_1 = tf.nn.relu(bn(conv3_1, is_training))
    pool3 = tf.layers.max_pooling2d(bn3_1, 2, 2)

    conv4 = dilated_conv2D_layer(pool3, 256, 'conv4')
    bn4 = tf.nn.relu(bn(conv4, is_training))
    conv4_1 = dilated_conv2D_layer(bn4, 256, 'conv4_1')
    bn4_1 = tf.nn.relu(bn(conv4_1, is_training))

    ap = aspp(bn4_1)

    up2 = tf.image.resize_images(ap, tf.shape(ap)[1:3] * 2)
    up2 = dilated_conv2D_layer(up2, 128, 'up2')
    up2 = attention(up2, bn3_1, is_training,128)
    up2 = dilated_conv2D_layer(up2, 128, 'up2_1')
    up2 = tf.nn.relu(bn(up2, is_training))
    up2 = dilated_conv2D_layer(up2, 128, 'up2_2')
    up2 = tf.nn.relu(bn(up2, is_training))

    up3 = tf.image.resize_images(up2, tf.shape(up2)[1:3] * 2)
    up3 = dilated_conv2D_layer(up3, 64, 'up3')
    up3 = attention(up3, bn2_1, is_training,64)
    up3 = dilated_conv2D_layer(up3, 64, 'up3_1')
    up3 = tf.nn.relu(bn(up3, is_training))
    up3 = dilated_conv2D_layer(up3, 64, 'up3_2')
    up3 = tf.nn.relu(bn(up3, is_training))

    up4 = tf.image.resize_images(up3, tf.shape(up3)[1:3] * 2)
    up4 = dilated_conv2D_layer(up4, 32, 'up4')
    up4 = attention(up4, bn1_1,is_training,32)
    up4 = dilated_conv2D_layer(up4, 32, 'up4_1')
    up4 = tf.nn.relu(bn(up4, is_training))
    up4 = dilated_conv2D_layer(up4, 32, 'up4_2')
    up4 = tf.nn.relu(bn(up4, is_training))

    logits = conv2D(up4, 1, 1)

    return logits