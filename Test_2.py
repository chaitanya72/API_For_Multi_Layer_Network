
import pytest
import numpy as np
from cnn import CNN
import os
import tensorflow
from tensorflow.keras.datasets import cifar10
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,MaxPool2D,InputLayer,Flatten
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import sparse_categorical_crossentropy,hinge,mean_squared_error
from tensorflow.keras.optimizers import Adam,RMSprop,SGD,Adagrad

def test_train1():
    from tensorflow.keras.datasets import cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train[:100,:]
    y_train= y_train[:100]
    #np.random.seed(100)
    initilizer = tensorflow.keras.initializers.Zeros()
    #initilizer = tensorflow.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=20)
    model_testing = CNN()
    model = Sequential()
    model.add(Conv2D(filters=64,kernel_size=3,strides=1,padding='same',activation='relu',trainable=True,input_shape=(32,32,3),kernel_initializer=initilizer,bias_initializer=initilizer))
    model.add(Conv2D(filters=70, kernel_size=3, strides=1, padding='same', activation='relu', trainable=True,kernel_initializer=initilizer,bias_initializer=initilizer))
    model.add(Conv2D(filters=75, kernel_size=3, strides=1, padding='same', activation='relu', trainable=True,kernel_initializer=initilizer,bias_initializer=initilizer))
    model.add(Conv2D(filters=90, kernel_size=3, strides=1, padding='same', activation='relu', trainable=True,kernel_initializer=initilizer,bias_initializer=initilizer))
    model.add(MaxPool2D(pool_size=2,padding='same',strides=1))
    model.add(Flatten())
    model.add(Dense(units=256,activation='relu',trainable=True,kernel_initializer=initilizer,bias_initializer=initilizer))
    model.add(Dense(units=256, activation='relu', trainable=True,kernel_initializer=initilizer,bias_initializer=initilizer))
    model.add(Dense(units=256, activation='sigmoid', trainable=True,kernel_initializer=initilizer,bias_initializer=initilizer))
    model.compile(optimizer='Adagrad',loss='hinge',metrics=['mse'])
    #history = model.fit(x=X_train,y=y_train,batch_size=32,epochs=5)
    #np.random.seed(100)
    model_testing.add_input_layer(shape=(32,32,3),name="")
    model_testing.append_conv2d_layer(num_of_filters=64,kernel_size=3,padding='same',strides=1,activation='relu',name="1")
    model_testing.append_conv2d_layer(num_of_filters=70, kernel_size=3, padding='same', strides=1, activation='relu',
                                      name="2")
    model_testing.append_conv2d_layer(num_of_filters=75, kernel_size=3, padding='same', strides=1, activation='relu',
                                      name="3")
    model_testing.append_conv2d_layer(num_of_filters=90, kernel_size=3, padding='same', strides=1, activation='relu',
                                      name="4")
    model_testing.append_maxpooling2d_layer(pool_size=2,padding='same',strides=1,name="5")
    model_testing.append_flatten_layer(name='6')
    model_testing.append_dense_layer(num_nodes=256,activation='relu',name='7')
    model_testing.append_dense_layer(num_nodes=256, activation='relu', name='8')
    model_testing.append_dense_layer(num_nodes=256, activation='sigmoid', name='9')
    model_testing.set_optimizer(optimizer='adagrad')
    model_testing.set_loss_function(loss='hinge')
    model_testing.set_metric(metric='mse')

    #model_testing.model.set_weights(model.get_weights())
    #print(model.get_weights())
    #assert all(model_testing.model.get_weights()[2]) == all(model.get_weights()[2])
    history = model.fit(x=X_train,y=y_train,batch_size=32,epochs=5,shuffle=False)
    loss = model_testing.train(X_train=X_train,y_train=y_train,batch_size=32,num_epochs=5)
    print(history.history.keys())
    #assert loss == history.history['loss']
    #assert model_testing.model.get_weights() == model.get_weights()
    #print(history.history.keys())
    #assertAlmostEqual(loss,history.history['loss'],delta=0.01)
    #assert pytest.approx(loss,0.1)==pytest.approx(history.history['loss'],0.1)
    assert np.allclose(loss,history.history['loss'], rtol=1e-2, atol=1e-2)
    #assert model_testing.model.get_weights() == model.get_weights()

def test_evaluate():
    from tensorflow.keras.datasets import cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train[:100, :]
    y_train = y_train[:100]
    X_test=X_test[:100,:]
    y_test = y_test[:100]
    # np.random.seed(100)
    initilizer = tensorflow.keras.initializers.Zeros()
    # initilizer = tensorflow.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=20)
    model_testing = CNN()
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', trainable=True,
                     input_shape=(32, 32, 3), kernel_initializer=initilizer, bias_initializer=initilizer))
    model.add(Conv2D(filters=70, kernel_size=3, strides=1, padding='same', activation='relu', trainable=True,
                     kernel_initializer=initilizer, bias_initializer=initilizer))
    model.add(Conv2D(filters=75, kernel_size=3, strides=1, padding='same', activation='relu', trainable=True,
                     kernel_initializer=initilizer, bias_initializer=initilizer))
    model.add(Conv2D(filters=90, kernel_size=3, strides=1, padding='same', activation='relu', trainable=True,
                     kernel_initializer=initilizer, bias_initializer=initilizer))
    model.add(MaxPool2D(pool_size=2, padding='same', strides=1))
    model.add(Flatten())
    model.add(
        Dense(units=256, activation='relu', trainable=True, kernel_initializer=initilizer, bias_initializer=initilizer))
    model.add(
        Dense(units=256, activation='relu', trainable=True, kernel_initializer=initilizer, bias_initializer=initilizer))
    model.add(Dense(units=256, activation='sigmoid', trainable=True, kernel_initializer=initilizer,
                    bias_initializer=initilizer))
    model.compile(optimizer='Adagrad', loss='hinge', metrics=['mse'])
    history = model.fit(x=X_train, y=y_train, batch_size=32, epochs=5, shuffle=False)





    model_testing.add_input_layer(shape=(32, 32, 3), name="")
    model_testing.append_conv2d_layer(num_of_filters=64, kernel_size=3, padding='same', strides=1, activation='relu',
                                      name="1")
    model_testing.append_conv2d_layer(num_of_filters=70, kernel_size=3, padding='same', strides=1, activation='relu',
                                      name="2")
    model_testing.append_conv2d_layer(num_of_filters=75, kernel_size=3, padding='same', strides=1, activation='relu',
                                      name="3")
    model_testing.append_conv2d_layer(num_of_filters=90, kernel_size=3, padding='same', strides=1, activation='relu',
                                      name="4")
    model_testing.append_maxpooling2d_layer(pool_size=2, padding='same', strides=1, name="5")
    model_testing.append_flatten_layer(name='6')
    model_testing.append_dense_layer(num_nodes=256, activation='relu', name='7')
    model_testing.append_dense_layer(num_nodes=256, activation='relu', name='8')
    model_testing.append_dense_layer(num_nodes=256, activation='sigmoid', name='9')
    model_testing.set_optimizer(optimizer='adagrad')
    model_testing.set_loss_function(loss='hinge')
    model_testing.set_metric(metric='mse')
    loss = model_testing.train(X_train=X_train, y_train=y_train, batch_size=32, num_epochs=5)

    model_evaluate = model.evaluate(X_test,y_test)

    model_testing_evaluate = model_testing.evaluate(X_test,y_test)

    #assert model_testing_evaluate == model_evaluate
    assert np.allclose(model_testing_evaluate,model_evaluate,rtol=1e-2,atol=1e-2)