import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and preprocess the data
print("Loading and preprocessing data...")
df = pd.read_csv('./model/spam_data.csv', encoding='latin-1')

text_column = 'v2'

# Check for missing values
print(f"Missing values in dataset: {df.isnull().sum().sum()}")

# Convert labels to binary format
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['v1'])  # 'ham' becomes 0, 'spam' becomes 1

label_classes = label_encoder.classes_
print(f"Label encoding: {dict(zip(label_classes, range(len(label_classes))))}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df[text_column].values, 
    df['label'].values, 
    test_size=0.2, 
    random_state=42
)

# Tokenize the text data
max_words = 10000  # Maximum number of words to keep based on frequency
max_sequence_length = 100  # Maximum length of sequences

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
print(f"Found {len(word_index)} unique tokens")

# Convert texts to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure uniform length
X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length)

print(f"Training data shape: {X_train_pad.shape}")
print(f"Testing data shape: {X_test_pad.shape}")

# Build the model
print("Building the model...")
embedding_dim = 100

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_words, embedding_dim, input_length=max_sequence_length),
    tf.keras.layers.SpatialDropout1D(0.2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# Train the model
print("Training the model...")
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train_pad, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Evaluate the model
print("Evaluating the model...")
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Save the model
print("Saving the model...")
model.export('./models/spam_detection_model')

# Save the tokenizer and label encoder for preprocessing new data
import pickle
with open('./model/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./model/label_encoder.pickle', 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Model, tokenizer, and label encoder saved successfully!")
