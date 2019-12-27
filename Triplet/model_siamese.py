import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, concatenate, BatchNormalization, PReLU, MaxPool2D, GlobalAveragePooling2D, Dropout, Dense, Lambda
from tensorflow.keras.regularizers import l2

def normalisation_layer(x):
    return(tf.nn.l2_normalize(x, 1, 1e-10))

def create_model(input_shape, drop_probability, bottleneck_layer_size=128, weight_decay=0.0):
    ip = Input(shape=input_shape)

    o_1_1 = Conv2D(32,(9,3),strides=1,padding='same',kernel_regularizer=l2(weight_decay))(ip)
    o_1_2 = Conv2D(32,(3,9),strides=1,padding='same',kernel_regularizer=l2(weight_decay))(ip)

    o_1 = concatenate([o_1_1,o_1_2])
    o_1 = MaxPool2D()(o_1)

    o_2_1 = Conv2D(64,(7,3),strides=1,padding='same',kernel_regularizer=l2(weight_decay))(o_1)
    o_2_2 = Conv2D(64,(3,7),strides=1,padding='same',kernel_regularizer=l2(weight_decay))(o_1)

    o_2 = concatenate([o_2_1,o_2_2])
    o_2 = PReLU()(o_2)
    o_2 = MaxPool2D((1,2))(o_2)

    o_3_1 = Conv2D(128,(5,3),strides=1,padding='same',kernel_regularizer=l2(weight_decay))(o_2)
    o_3_2 = Conv2D(128,(3,5),strides=1,padding='same',kernel_regularizer=l2(weight_decay))(o_2)

    o_3 = concatenate([o_3_1,o_3_2])
    o_3 = PReLU()(o_3)
    o_3 = MaxPool2D()(o_3)

    o_4 = Conv2D(128,3,strides=1,padding='same',kernel_regularizer=l2(weight_decay))(o_3)

    o_4 = PReLU()(o_4)
    o_4 = MaxPool2D()(o_4)

    o_5 = Conv2D(128,3,strides=1,padding='same',kernel_regularizer=l2(weight_decay))(o_4)
    o_5 = PReLU()(o_5)
    
    o_5 = Conv2D(128,3,strides=1,padding='same',kernel_regularizer=l2(weight_decay))(o_5)
    o_5 = PReLU()(o_5)

    o_5 = MaxPool2D()(o_5)

    o_6 = Conv2D(256,3,strides=1,padding='same',kernel_regularizer=l2(weight_decay))(o_5)
    o_6 = PReLU()(o_6)
    
    o_6 = Conv2D(512,3,strides=1,padding='same',kernel_regularizer=l2(weight_decay))(o_6)
    o_6 = PReLU()(o_6)

    o_6 = GlobalAveragePooling2D()(o_6)
    
    o_7 = Dropout(drop_probability)(o_6)
    o_7 = Dense(bottleneck_layer_size)(o_7)
    o = Lambda(normalisation_layer)(o_7)

    model = Model(ip,o)
    return model