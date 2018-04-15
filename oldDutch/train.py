import random
from keras.preprocessing import sequence
import tensorflow as tf
import numpy as np

Xdata = []
Ydata = []
MAX_LENGTH_WORD = 10

feature_dict = dict()
feature_list = list()

PADDING_CHARACTER = '~'
feature_dict[PADDING_CHARACTER] = 0
feature_list.append(PADDING_CHARACTER)
max_features = 1


def get_vector_from_string(input_s):
    global max_features
    vector_x = []
    for i in input_s:
        if i not in feature_dict:
            feature_dict[i] = max_features
            feature_list.append(i)
            max_features += 1
        vector_x.append(feature_dict[i])
    return vector_x


def add_to_data(input_s, output_s):
    if len(input_s) < MAX_LENGTH_WORD and len(output_s) < MAX_LENGTH_WORD:
        vector_x = get_vector_from_string(input_s)
        vector_y = get_vector_from_string(output_s)
        Xdata.append(vector_x)
        Ydata.append(vector_y)


def print_vector(vector, end_token='\n'):
    print(''.join([feature_list[i] for i in vector]), end=end_token)


with open("../input/dictionary_old_new_dutch.csv") as in_file:
    for line in in_file:
        in_s, out_s = line.strip().split(",")
        add_to_data(in_s, out_s)
for i in range(10):
    print_vector(Xdata[i], end_token='')
    print(' -> ', end='')
    print_vector(Ydata[i])

before_padding = Xdata[0]
Xdata = sequence.pad_sequences(Xdata, maxlen=MAX_LENGTH_WORD)
Ydata = sequence.pad_sequences(Ydata, maxlen=MAX_LENGTH_WORD)
after_padding = Xdata[0]

print_vector(before_padding, end_token='')
print(" -> after padding: ", end='')
print_vector(after_padding)


class DataSplitter:
    def __init__(self, percentage):
        self.percentage = percentage

    def split_data(self, data):
        splitpoint = int(len(data) * self.percentage)
        return data[:splitpoint], data[splitpoint:]


splitter = DataSplitter(0.8)
Xdata, Xtest = splitter.split_data(Xdata)
Ydata, Ytest = splitter.split_data(Ydata)


def get_random_reversed_dataset(Xdata, Ydata, batch_size):
    newX = []
    newY = []
    for _ in range(batch_size):
        index_taken = random.randint(0, len(Xdata) - 1)
        newX.append(Xdata[index_taken])
        newY.append(Ydata[index_taken][::-1])
    return newX, newY


batch_size = 64
memory_dim = 256
embedding_dim = 32

enc_input = [tf.placeholder(tf.int32, shape=(None,)) for i in range(MAX_LENGTH_WORD)]
dec_output = [tf.placeholder(tf.int32, shape=(None,)) for t in range(MAX_LENGTH_WORD)]

weights = [tf.ones_like(labels_t, dtype=tf.float32) for labels_t in enc_input]

dec_inp = ([tf.zeros_like(enc_input[0], dtype=np.int32)] + [dec_output[t] for t in range(MAX_LENGTH_WORD - 1)])
empty_dec_inp = ([tf.zeros_like(enc_input[0], dtype=np.int32, name="empty_dec_input") for t in range(MAX_LENGTH_WORD)])

setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)

lstm = tf.contrib.rnn.BasicLSTMCell(memory_dim)
# lstm = tf.contrib.rnn.MultiRNNCell([lstm]*3)

# cell = tf.contrib.rnn.GRUCell(memory_dim)
# import copy
# tmp_cell = copy.deepcopy(cell)

# Create a train version of encoder-decoder, and a test version which does not feed the previous input
with tf.variable_scope('decoder1', reuse=tf.AUTO_REUSE) as scope:
    output, _ = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(enc_input, dec_inp,
                                                                      lstm, num_encoder_symbols=max_features,
                                                                      num_decoder_symbols=max_features,
                                                                      embedding_size=memory_dim, feed_previous=False)
    scope.reuse_variables()
    runtime_output, _ = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(enc_input, empty_dec_inp,
                                                                              lstm, num_encoder_symbols=max_features,
                                                                              num_decoder_symbols=max_features,
                                                                              embedding_size=memory_dim,
                                                                              feed_previous=True)

loss = tf.contrib.legacy_seq2seq.sequence_loss(output, dec_output, weights, max_features)

optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

# Init everything
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for index_now in range(3002):
    Xin, Yin = get_random_reversed_dataset(Xdata, Ydata, batch_size)
    Xin = np.array(Xin).T
    Yin = np.array(Yin).T
    feed_dict = {enc_input[t]: Xin[t] for t in range(MAX_LENGTH_WORD)}
    feed_dict.update({dec_output[t]: Yin[t] for t in range(MAX_LENGTH_WORD)})
    _, l = sess.run([train_op, loss], feed_dict)
    if index_now % 100 == 1:
        print(l)


def get_reversed_max_string_logits(logits):
    string_logits = logits[::-1]
    concatenated_string = ""
    for logit in string_logits:
        val_here = np.argmax(logit)
        concatenated_string += feature_list[val_here]
    return concatenated_string


def print_out(out):
    out = list(zip(*out))
    out = out[:10]  # only show the first 10 samples

    for index, string_logits in enumerate(out):
        print("input: ", end='')
        print_vector(Xin[index])
        print("expected: ", end='')
        expected = Yin[index][::-1]
        print_vector(expected)

        output = get_reversed_max_string_logits(string_logits)
        print("output: " + output)

        print("==============")


# Now run a small test to see what our network does with words
RANDOM_TESTSIZE = 5
Xin, Yin = get_random_reversed_dataset(Xtest, Ytest, RANDOM_TESTSIZE)
Xin_transposed = np.array(Xin).T
Yin_transposed = np.array(Yin).T
feed_dict = {enc_input[t]: Xin_transposed[t] for t in range(MAX_LENGTH_WORD)}
out = sess.run(runtime_output, feed_dict)
print_out(out)


def translate_single_word(word):
    Xin = [get_vector_from_string(word)]
    Xin = sequence.pad_sequences(Xin, maxlen=MAX_LENGTH_WORD)
    Xin_transposed = np.array(Xin).T
    feed_dict = {enc_input[t]: Xin_transposed[t] for t in range(MAX_LENGTH_WORD)}
    out = sess.run(runtime_output, feed_dict)
    return get_reversed_max_string_logits(out)


interesting_words = ["aerde", "duyster", "salfde", "ontstondt", "tusschen", "wacker", "voorraet", "gevreeset",
                     "cleopatra"]
for word in interesting_words:
    print(word + " becomes: " + translate_single_word(word).replace("~", ""))
