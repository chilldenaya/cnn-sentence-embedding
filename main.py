import tensorflow as tf
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Embedding
from tensorflow.keras.models import Sequential

# The inputs are 32-length word vectors with 20 word per sentence, and the
# dataset size is 100.
sentence_max_length = 20
word_embedding_length = 32
dataset_length = 100
input_shape = (dataset_length, sentence_max_length, word_embedding_length)
x = tf.random.normal(input_shape)

vocab_size = 1000
sentence_embedding_length = 3
kernel_size = 3

model = Sequential()
model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=word_embedding_length,
        input_length=sentence_max_length,
    )
)
model.add(
    Conv1D(
        filters=sentence_embedding_length,
        kernel_size=kernel_size,
        activation="relu",
        strides=1,
        padding="valid",
    )
)
model.add(GlobalMaxPooling1D())

# Output shape: (None, (max_sent_length-kernel_size+1), sentence_embedding_length)

model.build(input_shape)
print(model.summary())
