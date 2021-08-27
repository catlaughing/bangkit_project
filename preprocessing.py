import numpy as np
from tensorflow.python.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, RepeatVector, Embedding, LSTM, TimeDistributed, Concatenate, Activation
from inference import read_helper_file


def preprocessing_image(img_path):
    im = image.load_img(img_path, target_size=(224, 224, 3))
    im = image.img_to_array(im)
    im = np.expand_dims(im, axis=0)
    im = preprocess_input(im)
    return im


def get_encoding(model, img):
    pred = model.predict(img).reshape((1,2048))
    return pred


# def create_model():
#     embedding_size = 128
#     max_len = 40
#     word_index,_ = read_helper_file()
#     vocab_size = len(word_index)
#
#     # Image Input
#     image_model = Sequential()
#     image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
#     image_model.add(RepeatVector(max_len))
#
#     # Caption Input
#     language_model = Sequential()
#     language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
#     language_model.add(LSTM(256, return_sequences=True))
#     language_model.add(TimeDistributed(Dense(embedding_size)))
#
#     conca = Concatenate()([image_model.output, language_model.output])
#     x = LSTM(128, return_sequences=True)(conca)
#     x = LSTM(512, return_sequences=False)(x)
#     x = Dense(vocab_size)(x)
#     out = Activation('softmax')(x)
#     model = Model(inputs=[image_model.input, language_model.input], outputs=out)
#     model.load_weights('models/model_weights.h5')
#
#     return model