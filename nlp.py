import tensorflow as tf
import tensorflow_datasets as tfds
import os
	
import datetime

parent_dir = "/Users/erdemisbilen/TFvenv/"
FILE_NAMES = ['article_all.txt']

BUFFER_SIZE = 2000
BATCH_SIZE = 128
TAKE_SIZE = 20000

def labeler(example, index):
  return example, tf.cast(index, tf.int64)  

labeled_data_sets = []

for i, file_name in enumerate(FILE_NAMES):
  lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
  labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
  labeled_data_sets.append(labeled_dataset)

all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
  all_labeled_data = all_labeled_data.concatenate(labeled_dataset)
  
all_labeled_data = all_labeled_data.shuffle(
    BUFFER_SIZE, reshuffle_each_iteration=False)

print("Dataset Items Before Encoding:")
print("-------------------------------")
for ex in all_labeled_data.take(2):
  print(ex)
print("-------------------------------")

tokenizer = tfds.features.text.Tokenizer()

vocabulary_set = set()

for text_tensor, _ in all_labeled_data:
  some_tokens = tokenizer.tokenize(text_tensor.numpy())
  vocabulary_set.update(some_tokens)

vocab_size = len(vocabulary_set)

print("Vocabulary size.   :" + str(vocab_size))
print("-------------------------------")
print(vocabulary_set)
print("-------------------------------")

encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

example_text = next(iter(all_labeled_data))[0].numpy()
print("Example Sentence:")
print("-------------------------------")
print(example_text)

print("Encoded Example Sentence:")
print("-------------------------------")
encoded_example = encoder.encode(example_text)
print(encoded_example)

def encode(text_tensor, label):
  encoded_text = encoder.encode(text_tensor.numpy())
  return encoded_text, label

def encode_map_fn(text, label):
  encoded_text, label = tf.py_function(encode, 
                                       inp=[text, label], 
                                       Tout=(tf.int64, tf.int64))
  return encoded_text, label

all_encoded_data = all_labeled_data.map(encode_map_fn)



train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([200],()))

test_data = all_encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([200],()))


sample_text, sample_labels = next(iter(test_data))

print(sample_text[10])
print(sample_labels[10])

print(sample_text[11])
print(sample_labels[11])

#Training a LSTM model to test the data pipeline
vocab_size += 1

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 64))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))


for units in [64, 64]:
  model.add(tf.keras.layers.Dense(units, activation='relu'))

# Output layer. The first argument is the number of labels.
model.add(tf.keras.layers.Dense(3, activation='softmax'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.005, amsgrad=True)


model.compile(optimizer= optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


model.fit(train_data, epochs=10, steps_per_epoch=4, validation_data=test_data, callbacks=[tensorboard_callback])

eval_loss, eval_acc = model.evaluate(test_data)

print('\nEval loss: {:.3f}, Eval accuracy: {:.3f}'.format(eval_loss, eval_acc))
