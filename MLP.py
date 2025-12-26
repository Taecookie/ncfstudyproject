'''
MLP_TF2.py
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import os

def get_model(num_users, num_items, layers=[20,10], reg_layers=[0,0]):
    num_layer = len(layers)
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=int(layers[0]/2), name='user_embedding',
                                   embeddings_initializer='random_normal', embeddings_regularizer=l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=int(layers[0]/2), name='item_embedding',
                                   embeddings_initializer='random_normal', embeddings_regularizer=l2(reg_layers[0]), input_length=1)
    
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))
    
    vector = Concatenate()([user_latent, item_latent])
    
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers[idx]), activation='relu', name='layer%d' %idx)
        vector = layer(vector)
        
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(vector)
    
    model = Model(inputs=[user_input, item_input], outputs=prediction)
    return model

def get_train_instances(train, num_negatives, num_items):
    user_input, item_input, labels = [],[],[]
    for (u, i) in train.keys():
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

if __name__ == '__main__':
    if not os.path.exists('Pretrain'):
        os.makedirs('Pretrain')
        
    path = 'Data/'
    dataset_name = 'ml-1m'
    epochs = 20
    batch_size = 256
    layers = [64,32,16,8]
    reg_layers = [0,0,0,0]
    num_negatives = 4
    learning_rate = 0.001
    
    dataset = Dataset(path + dataset_name)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("MLP: Load data done.")
    
    model = get_model(num_users, num_items, layers, reg_layers)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy')
    
    for epoch in range(epochs):
        t1 = time()
        user_input, item_input, labels = get_train_instances(train, num_negatives, num_items)
        hist = model.fit([np.array(user_input), np.array(item_input)], np.array(labels), 
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        
        if epoch % 5 == 0:
            print('Iteration %d: loss = %.4f [%.1f s]' % (epoch, hist.history['loss'][0], time()-t1))

    model.save_weights('Pretrain/MLP_weights.h5')
    print("MLP weights saved to Pretrain/MLP_weights.h5")