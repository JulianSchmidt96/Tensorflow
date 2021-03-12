import functools
import numpy as np
import tensorflow as tf
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

np.set_printoptions(precision=3, suppress=True)   #textgroeße lessbarer machen

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)


csv_file_train=tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)

df = pd.read_csv(csv_file_train)

df.head()


LABEL_COLUMN = 'survived'
LABELS = [0, 1]



def get_dataset(file_path, **kwargs):
  dataset = tf.data.experimental.make_csv_dataset(  #csv zu dataset
      file_path,
      batch_size=25, # klein, zur einfachreren Anzeige
      label_name=LABEL_COLUMN, #Label_Column wird als separater Vektor abgeschnitten
      na_value="?",#zur Kennzeichnung von as NA/NaN.
      num_epochs=1,
      ignore_errors=True, 
      **kwargs)#eintragen der params an richtige stelle durch kwargs zuordnung
  return dataset

raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)


def show_batch(dataset):
  for batch, label in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))

#show_batch(raw_train_data)




SELECT_COLUMNS = ['survived', 'age', 'n_siblings_spouses', 'parch', 'fare']  
DEFAULTS = [0, 0.0, 0.0, 0.0, 0.0]
temp_dataset = get_dataset(train_file_path,     #csv in richtiges dateiformat, falls in einzelnen spalten,zeilen verschied datatypes
                           select_columns=SELECT_COLUMNS,
                           column_defaults = DEFAULTS)

show_batch(temp_dataset)


example_batch, labels_batch = next(iter(temp_dataset)) 

def pack(features, label):
     return tf.stack(list(features.values()), axis=-1), label  #Spalten zsm packen

packed_dataset = temp_dataset.map(pack)

for features, labels in packed_dataset.take(1):#Spalten zsm packen
    print(features.numpy())
    print()
    print(labels.numpy())





class PackNumericFeatures(object): #alle  features in eine Spalte pruegeln   -> ein vektor fuer z.B. age enthaelt alle Altersangaben
  def __init__(self, names):
    self.names = names

  def __call__(self, features, labels):
    numeric_features = [features.pop(name) for name in self.names]
    numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
    numeric_features = tf.stack(numeric_features, axis=-1)
    features['numeric'] = numeric_features

    return features, labels

NUMERIC_FEATURES = ['age','n_siblings_spouses','parch', 'fare']

packed_train_data = raw_train_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))

packed_test_data = raw_test_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))

#show_batch(packed_train_data)

example_batch, labels_batch = next(iter(packed_train_data)) 


desc = pd.read_csv(train_file_path)[NUMERIC_FEATURES].describe()
desc


MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])

def normalize_numeric_data(data, mean, std): #normierung durch Standardabweichung
  return (data-mean)/std






normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)#mean und std an normalize_numeric_data binden

numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]
numeric_column

example_batch['numeric']



numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns) #erste Schicht bilden-> Numerische Unterscheidung
numeric_layer(example_batch).numpy()


CATEGORIES = { # Entscheidungskategorien nennen
    'sex': ['male', 'female'],
    'class' : ['First', 'Second', 'Third'],
    'deck' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town' : ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone' : ['y', 'n']
}

categorical_columns = []
for feature, vocab in CATEGORIES.items():
  cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
  categorical_columns.append(tf.feature_column.indicator_column(cat_col))
#print(categorical_columns)


categorical_layer = tf.keras.layers.DenseFeatures(categorical_columns)  #Schicht en um kategorienunterscheidung erweitern
#print(categorical_layer(example_batch).numpy()[0])

preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numeric_columns) #Input Schicht  soll ein array sein→ um Schwellenwert der Aktivierungsfkt zu testen

#print(preprocessing_layer(example_batch).numpy()[0])


model = tf.keras.Sequential([ #Model mit Kearas Sequential-> Sequential fuehrt Training und Interferenzen ein
  preprocessing_layer,
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1),
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])



train_data = packed_train_data.shuffle(500) #Zufallsanordnung
test_data = packed_test_data

model.fit(train_data, epochs=20)     #Training

test_loss, test_accuracy = model.evaluate(test_data)

print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy)) #Auswertung

predictions = model.predict(test_data)



for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]): #Abschaetzung der Ouputs fuer Input Samples
  prediction = tf.sigmoid(prediction).numpy()
  print("Predicted survival: {:.2%}".format(prediction[0]),
        " | Actual outcome: ",
        ("SURVIVED" if bool(survived) else "DIED"))

