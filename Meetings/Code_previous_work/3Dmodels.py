import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import Input, Dense, Dropout, Conv3D, Conv3DTranspose, UpSampling3D, MaxPooling3D, MaxPool3D, Flatten, BatchNormalization, Lambda
from tensorflow.keras.models import Model
import h5py


##choose type of model to run
model_to_use ='simple'  #'vgg16'
validation_rounds = 50

##load data
##nifti layers and labels
with h5py.File('train_data_cat4_all.h5', 'r') as hf:
    train_data = hf['norm_data'][...]
    labels = hf['lbl_data'][...]

print(train_data.shape)
print(labels.shape)

with h5py.File('test_data_cat4_v2.h5', 'r') as hf:
    test_data = hf['norm_data'][...]
    test_labels = hf['lbl_data'][...]
    
print(test_data.shape)
print(test_labels.shape)



def create_VGG_s_3Dmodel(dim_x, dim_y, dim_z):
    # input layer
    input_layer = Input((dim_x, dim_y, dim_z, 3))

    # convolutional layers
    conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu')(input_layer)
    conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu')(conv_layer1)

    # add max pooling
    pooling_layer1 = MaxPool3D(pool_size=(3, 3, 3))(conv_layer2)
    pooling_layer1 = BatchNormalization()(pooling_layer1)

    conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(pooling_layer1)
    conv_layer4 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(conv_layer3)
    pooling_layer2 = MaxPool3D(pool_size=(3, 3, 3))(conv_layer4)

    # perform batch normalization on the convolution outputs before feeding it to MLP architecture
    pooling_layer2 = BatchNormalization()(pooling_layer2)
    flatten_layer = Flatten()(pooling_layer2)

    # create an MLP architecture with dense layers
    # add dropouts to avoid overfitting / perform regularization
    dense_layer1 = Dense(units=4096, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.5)(dense_layer1)
    dense_layer2 = Dense(units=1024, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.5)(dense_layer2)
    output_layer = Dense(units=4, activation='softmax')(dense_layer2)

    # define the model with input layer and output layer
    model = Model(inputs=input_layer, outputs=output_layer)

    return model


def create_VGG_16_3Dmodel(dim_x, dim_y, dim_z):
    input_layer = Input((dim_x, dim_y, dim_z, 3))

    conv_layer1_1 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(input_layer)
    conv_layer1_2 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(conv_layer1_1)
    pooling_layer1 = MaxPool3D(pool_size=(2, 2, 1))(conv_layer1_2)
    pooling_layer1 = BatchNormalization()(pooling_layer1)

    conv_layer2_1 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu')(pooling_layer1)
    conv_layer2_2 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu')(conv_layer2_1)
    pooling_layer2 = MaxPool3D(pool_size=(2, 2, 1))(conv_layer2_2)
    pooling_layer2 = BatchNormalization()(pooling_layer2)

    conv_layer3_1 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu')(pooling_layer2)
    conv_layer3_2 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu')(conv_layer3_1)
    conv_layer3_3 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu')(conv_layer3_2)
    pooling_layer3 = MaxPool3D(pool_size=(2, 2, 1))(conv_layer3_3)
    pooling_layer3 = BatchNormalization()(pooling_layer3)

    # perform batch normalization on the convolution outputs before feeding it to MLP architecture
    flatten_layer = Flatten()(pooling_layer3)

    # create an MLP architecture with dense layers
    # add dropouts to avoid overfitting / perform regularization
    dense_layer1 = Dense(units=1024, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.5)(dense_layer1)
    dense_layer2 = Dense(units=512, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.5)(dense_layer2)
    output_layer = Dense(units=4, activation='softmax')(dense_layer2)

    # define the model with input layer and output layer
    model = Model(inputs=input_layer, outputs=output_layer)

    return model


out_file_name = 'cat4_simple_model_acc.out'
out_file_name_lbls = 'cat4_simple_model_lbl.out'

if model_to_use == 'vgg16':
    out_file_name = 'cat4_vgg16_model_acc.out'
    out_file_name_lbls = 'cat4_vgg16_model_lbl.out'

d = open(out_file_name, "w+")
d_lbl = open(out_file_name_lbls, "w+")


for k_round in range (0, validation_rounds):
    d.write('Evaluating model at round ' + str(k_round+1) + '\n')
    lrn_acc = 0
    tst_acc = 0
    
    ytest = tf.keras.utils.to_categorical(test_labels, 4)
    ytrain = tf.keras.utils.to_categorical(labels, 4)

    model = create_VGG_s_3Dmodel(95, 79, 60)
    if model_to_use == 'vgg16':
        model = create_VGG_16_3Dmodel(95, 79, 60)

    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adadelta(lr=0.01),metrics=['acc'])
    history = model.fit(x=train_data, y=ytrain, batch_size=6, epochs=50)
    #train_dataset = tf.data.Dataset.from_tensor_slices((train_data, ytrain)).batch(6)
    #history = model.fit(train_dataset, epochs=10, batch_size=6)
    accuracy_history = history.history['acc']
    lrn_acc = accuracy_history[-1]

    #test_set = tf.data.Dataset.from_tensor_slices((test_data, ytest)).batch(6)
    test_loss, tst_acc = model.evaluate(test_data, ytest, batch_size=6)
    print('testing acc. ' + str(tst_acc))

    if tst_acc > 0.7:
        model_name = "mcat4_all_" + model_to_use +  '_round' + str(k_round + 1) + ".h5"
        model.save(model_name)
        d.write("Saved model to disk" + "  " + model_name + '\n')

    d.write('Finished for round ' + str(k_round+1) + '\n')
    d.write('training acc.:' + str(lrn_acc) + '\n')
    d.write('test acc.:' + str(tst_acc) + '\n')
    d.flush()

    #geting predicted labels
    d_lbl.write('round: '+ str(k_round + 1)+ '\n')
    d_lbl.write('\n')
    d_lbl.write('test_predict_lbl:' + '\n' ) 
    ypred = model.predict(test_data, batch_size=6)
    for yp in ypred:
        d_lbl.write(str(np.argmax(yp)) + ' ')
    
    d_lbl.write('\n')
    d_lbl.flush()


d.close()
d_lbl.close()
