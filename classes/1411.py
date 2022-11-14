import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

import wandb
from wandb.keras import WandbCallback

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

train_x, test_x = map(lambda x: x[..., None], [train_x, test_x])

encoder = OneHotEncoder(categories=[range(10)])

encoded_train_y, encoded_test_y = map(
    lambda y: encoder.fit_transform(y.reshape(-1, 1)).toarray(), [train_y, test_y]
) 

model = tf.keras.models.Sequential([
    tf.keras.Input(train_x.shape[1:]),
    tf.keras.layers.Conv2D(5, (3, 3), activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(3, (3, 3), activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(encoded_train_y.shape[-1], activation="softmax")
    ])

optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

wandb.init(project="asi-classes")


model.fit(train_x, encoded_train_y, batch_size=120,
 epochs=5, callbacks=[WandbCallback()], validation_data=(test_x, encoded_test_y),
 validation_batch_size=128,)

