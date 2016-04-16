import sys
from kraino.utils import data_provider

def print_list(ll):
    # Prints the list
    print('\n'.join(ll))

dp = data_provider.select['daquar-triples']
train_text_representation = dp['text'](train_or_test='train')

n_elements = 10
#print('== Questions:')
#print_list(train_text_representation['x'][:n_elements])
#print('== Answers:')
#print_list(train_text_representation['y'][:n_elements])
#print('== Image Names:')
#print_list(train_text_representation['img_name'][:n_elements])

from toolz import frequencies
train_raw_x = train_text_representation['x']
# we start from building the frequencies table
wordcount_x = frequencies(' '.join(train_raw_x).split(' '))
# print the most and least frequent words
n_show = 5
#print(sorted(wordcount_x.items(), key=lambda x: x[1], reverse=True)[:n_show])
#print(sorted(wordcount_x.items(), key=lambda x: x[1])[:n_show])

# Kraino is a framework that helps in fast prototyping Visual Turing Test models
from kraino.utils.input_output_space import build_vocabulary

# This function takes wordcounts and returns word2index - mapping from words into indices,
# and index2word - mapping from indices to words.
word2index_x, index2word_x = build_vocabulary(
    this_wordcount=wordcount_x,
    truncate_to_most_frequent=0)

#print (word2index_x)
#print(sorted(word2index_x, key=lambda x: word2index_x[x]))


from kraino.utils.input_output_space import encode_questions_index
one_hot_x = encode_questions_index(train_raw_x, word2index_x)
#print(train_raw_x[:3])
#print(one_hot_x[:3])

from keras.preprocessing import sequence
MAXLEN=30
train_x = sequence.pad_sequences(one_hot_x, maxlen=MAXLEN)
#print(train_x[:3])

MAX_ANSWER_TIME_STEPS=1

from kraino.utils.input_output_space import encode_answers_one_hot
train_raw_y = train_text_representation['y']
wordcount_y = frequencies(' '.join(train_raw_y).split(' '))
word2index_y, index2word_y = build_vocabulary(this_wordcount=wordcount_y)
train_y, _ = encode_answers_one_hot(
    train_raw_y,
    word2index_y,
    answer_words_delimiter=train_text_representation['answer_words_delimiter'],
    is_only_first_answer_word=True,
    max_answer_time_steps=MAX_ANSWER_TIME_STEPS)
print(train_x.shape)
print(train_y.shape)

### TEST SET LOADING STARTS HERE ###


test_text_representation = dp['text'](train_or_test='test')
test_raw_x = test_text_representation['x']
test_one_hot_x = encode_questions_index(test_raw_x, word2index_x)
test_x = sequence.pad_sequences(test_one_hot_x, maxlen=MAXLEN)

test_raw_y = test_text_representation['y']
test_y, _ = encode_answers_one_hot(
    test_raw_y, 
    word2index_y, 
    answer_words_delimiter=test_text_representation['answer_words_delimiter'],
    is_only_first_answer_word=True,
    max_answer_time_steps=MAX_ANSWER_TIME_STEPS)
print(test_x.shape)
print(test_y.shape)

test_image_names = test_text_representation['img_name']

test_visual_features = dp['perception'](
    train_or_test='test',
    names_list=test_image_names,
    parts_extractor=None,
    max_parts=None,
    perception=CNN_NAME,
    layer=PERCEPTION_LAYER,
    second_layer=None,
)

### TEST SET LOADING ENDS HERE ###

# this contains a list of the image names of our interest;
# it also makes sure that visual and textual features are aligned correspondingly
train_image_names = train_text_representation['img_name']
# the name for visual features that we use
# CNN_NAME='vgg_net'
CNN_NAME='vgg_net'
# the layer in CNN that is used to extract features
# PERCEPTION_LAYER='fc7'
PERCEPTION_LAYER='fc7'

train_visual_features = dp['perception'](
    train_or_test='train',
    names_list=train_image_names,
    parts_extractor=None,
    max_parts=None,
    perception=CNN_NAME,
    layer=PERCEPTION_LAYER,
    second_layer=None
    )
print(train_visual_features.shape)





print(len(word2index_x))
print(len(word2index_y))

print(word2index_y)

import keras
from keras.models import Graph
from keras.layers import Dense, Dropout, Embedding, LSTM, Activation, Merge

model = keras.models.Graph()
model.add_input(name='langInput', input_shape=(MAXLEN,), dtype='int')
model.add_node(Embedding(input_dim=len(word2index_x), input_length=MAXLEN, output_dim=512), input='langInput', name='langEmbedding')
model.add_node(LSTM(output_dim=512, activation='sigmoid', inner_activation='hard_sigmoid'), input='langEmbedding', name='langLSTM')
model.add_node(Dropout(0.5), input='langLSTM', name='langDropout')

model.add_input(name='imInput', input_shape=(train_visual_features.shape[1],))
model.add_node(Dense(output_dim=512), input='imInput', name='imDense')
model.add_node(Dropout(0.5), input='imDense', name='imDropout')
model.add_node(Activation(activation='tanh'), input='imDropout', name='imActivation')

model.add_node(Activation(activation='linear'), inputs=['langDropout','imActivation'], merge_mode='sum', name='merged')
model.add_node(Dense(output_dim=len(word2index_y)), input='merged', name='mergeDense')
model.add_node(Activation(activation='softmax'), input='mergeDense', name='softmax')
model.add_output(name='output', input='softmax')

print("compiling model")

from keras.optimizers import Adam
model.compile(loss={'output': 'categorical_crossentropy'}, optimizer=Adam(lr=0.001), )

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=100)


train_input = [train_x, train_visual_features]
print(model.__class__)
model.fit(  {'imInput': train_visual_features, 'langInput': train_x, 'output': train_y},
            nb_epoch=100,
            verbose=1,
            batch_size=512,
            validation_split=0.1,
            show_accuracy=True,
            callbacks=[early_stopping])


testYtmp = model.predict({'imInput': test_visual_features, 'langInput': test_x})

testY = testYtmp['output']

print testY.shape
print test_y.shape


import numpy as np

errors = 0
for i in range(testY.shape[0]):
    trueAnswerIdx = np.argmax(test_y[i, :])
    modelAnswerIdx = np.argmax(testY[i, :])
    if (trueAnswerIdx != modelAnswerIdx):
        errors += 1 


accuracy = 1. - (float(errors)/testY.shape[0])

print(accuracy)
