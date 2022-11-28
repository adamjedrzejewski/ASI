import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

import optuna
from optuna.integration import KerasPruningCallback

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

train_x, test_x = map(lambda x: x[..., None], [train_x, test_x])

encoder = OneHotEncoder(categories=[range(10)])
encoded_train_y, encoded_test_y = map(
    lambda y: encoder.fit_transform(y.reshape(-1, 1)).toarray(), [train_y, test_y]
)


def create_model(trial: optuna.Trial) -> tf.keras.models.Model:
    return tf.keras.models.Sequential(
    [
        tf.keras.layers.Input(train_x.shape[1:]),
        tf.keras.layers.Conv2D(5, (3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(3, (3, 3), activation="relu"),
        tf.keras.layers.Flatten(), 
        *(
            tf.keras.layers.Dense(
                                  trial.suggest_int(name=f"dense-neurons-{i}", low=4, high=8),
                                  activation="relu")
            for i in range(trial.suggest_int(name="denses-count", low=1, high=5))
        ),
        tf.keras.layers.Dense(12, activation="relu"),
        tf.keras.layers.Dense(encoded_train_y.shape[-1], activation="softmax"),
    ]
)
def objective(trial: optuna.Trial) -> float:
    model = create_model(trial)
    optimizer = tf.keras.optimizers.Adam(learning_rate=10**-5)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(
        train_x,
        encoded_train_y,
        batch_size=128,
        epochs=5,
        callbacks=[],
        validation_data=(test_x, encoded_test_y),
        validation_batch_size=128,
    )

    optimizer.learning_rate = 10**-3

    loss, accuracy = model.evaluate(test_x, encoded_test_y, batch_size=128)

    return accuracy

study = optuna.create_study(
    storage="sqlite:///nn-trials.db",
    sampler=optuna.samplers.NSGAIISampler(),
    pruner=optuna.pruners.SuccessiveHalvingPruner(),
    study_name="my-test-study-three",
    load_if_exists=True,
    direction=optuna.study.StudyDirection.MAXIMIZE
)

study.optimize(
    func=objective,
    n_jobs=1,
    n_trials=3,
    show_progress_bar=True,
)
