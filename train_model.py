import tensorflow as tf
from tensorflow.keras.layers import Embedding, MultiHeadAttention, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys

# Constants
EMBEDDING_DIM = 512
VOCAB_SIZE = 50000
MAX_SEQUENCE_LENGTH = 256

# Tokenization and sequence conversion
def preprocess_data(text):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
    tokenizer.fit_on_texts([text])

    sequences = tokenizer.texts_to_sequences([text])[0]

    input_sequences = []
    for i in range(1, len(sequences)):
        n_gram_sequence = sequences[max(0, i - MAX_SEQUENCE_LENGTH):i+1]
        input_sequences.append(n_gram_sequence)

    padded_sequences = pad_sequences(input_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='pre')
    X = padded_sequences[:, :-1]
    y = padded_sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=VOCAB_SIZE)
    
    return X, y

# Model definition
def create_model():
    inputs = Input(shape=(None,))
    embedding_layer = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(inputs)
    transformer_block = MultiHeadAttention(num_heads=8, key_dim=EMBEDDING_DIM)(embedding_layer, embedding_layer)
    outputs = Dense(VOCAB_SIZE, activation='softmax')(transformer_block)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Main training function
def train(text):
    X, y = preprocess_data(text)
    model = create_model()
    model.fit(X, y, epochs=10, batch_size=64, validation_split=0.2)
    model.save('/app/models/trained_model.h5')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please provide the path to the text file for training.")
        sys.exit(1)

    with open(sys.argv[1], 'r') as file:
        data = file.read()
        train(data)
