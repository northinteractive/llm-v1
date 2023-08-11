from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.layers import Embedding, MultiHeadAttention, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Constants
EMBEDDING_DIM = 512
VOCAB_SIZE = 50000
MAX_SEQUENCE_LENGTH = 256
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')

def preprocess_data(text):
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

def create_model():
    inputs = Input(shape=(None,))
    embedding_layer = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(inputs)
    transformer_block = MultiHeadAttention(num_heads=8, key_dim=EMBEDDING_DIM)(embedding_layer, embedding_layer)
    outputs = Dense(VOCAB_SIZE, activation='softmax')(transformer_block)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

@app.route('/train', methods=['POST'])
def train_endpoint():
    raw_data = request.data.decode('utf-8')

    X, y = preprocess_data(raw_data)
    model = create_model()
    model.fit(X, y, epochs=10, batch_size=64, validation_split=0.2)
    model.save('/app/models/trained_model.h5')

    return jsonify({"message": "Training successful"}), 200

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    model = create_model()
    model.load_weights('/app/models/trained_model.h5')

    data = request.data.decode("utf-8")
    sequences = tokenizer.texts_to_sequences([data])
    padded_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH - 1, padding='pre')
    
    predicted = model.predict(padded_data)
    predicted_word_index = tf.argmax(predicted, axis=-1).numpy()[0]
    predicted_word = tokenizer.index_word.get(predicted_word_index, 'Unknown')

    return jsonify({"predicted_word": predicted_word}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)