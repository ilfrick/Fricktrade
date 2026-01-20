import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

INSAMPLES = 12
OUTSAMPLES = 12
N_FEATURES = 5
PRICE_COLS = slice(0, 4)
TARGET_COL = 3
VOLUME_COL = 4
EPS = 1e-6

BATCH_SIZE = 512
EPOCHS = 20
LSTM_UNITS = 128
DENSE_UNITS = 128

TRAIN_FILES = tf.io.gfile.glob(
    "/clamfs/Shared/Documents/ML/stocks_v6_yfinance/stocks*train*"
)
VAL_FILES = tf.io.gfile.glob(
    "/clamfs/Shared/Documents/ML/stocks_v6_yfinance/stocks*test*"
)

SAVE_FILE = (
    f"stocks_price_regression_{LSTM_UNITS}_"
    f"{INSAMPLES * 5}-{OUTSAMPLES * 5}m_5minc"
)

AUTOTUNE = tf.data.AUTOTUNE


def parse_tfrecord_fn(serialized_example):
    feature_description = {
        "x": tf.io.FixedLenFeature([], tf.string),
        "y": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    example["x"] = tf.io.parse_tensor(example["x"], out_type=tf.float32)
    example["y"] = tf.io.parse_tensor(example["y"], out_type=tf.float32)
    return example


def _log_returns(values):
    prev = values[:-1]
    curr = values[1:]
    log_ret = tf.math.log(curr + EPS) - tf.math.log(prev + EPS)
    first = tf.zeros_like(log_ret[:1])
    return tf.concat([first, log_ret], axis=0)


def prepare_sample(features):
    x = features["x"]
    y = features["y"]
    x = x[-INSAMPLES:, :]
    y = y[:OUTSAMPLES, :]

    price = x[:, PRICE_COLS]
    price_feat = _log_returns(price)

    volume = x[:, VOLUME_COL:VOLUME_COL + 1]
    volume = tf.math.log1p(volume)
    vol_mean = tf.reduce_mean(volume, axis=0, keepdims=True)
    vol_std = tf.math.reduce_std(volume, axis=0, keepdims=True)
    volume = tf.math.divide_no_nan(volume - vol_mean, vol_std + EPS)

    x_feat = tf.concat([price_feat, volume], axis=1)
    x_feat = tf.ensure_shape(x_feat, [INSAMPLES, N_FEATURES])

    last_close = x[-1, TARGET_COL]
    y_close = y[:, TARGET_COL]
    y_log_return = tf.math.log(y_close + EPS) - tf.math.log(last_close + EPS)
    y_log_return = tf.ensure_shape(y_log_return, [OUTSAMPLES])

    return x_feat, y_log_return


def get_dataset(filenames, batch_size, shuffle=True):
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(prepare_sample, num_parallel_calls=AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(AUTOTUNE)


def build_model():
    inputs = keras.Input(shape=(INSAMPLES, N_FEATURES), name="x")
    x = layers.LayerNormalization()(inputs)
    x = layers.LSTM(LSTM_UNITS, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(LSTM_UNITS)(x)
    x = layers.Dense(DENSE_UNITS, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(OUTSAMPLES, activation="linear")(x)
    return keras.Model(inputs, outputs)


def _features_from_window(window):
    window = tf.convert_to_tensor(window, dtype=tf.float32)
    window = window[:, :N_FEATURES]
    if window.shape[0] != INSAMPLES:
        raise ValueError(f"Expected {INSAMPLES} rows, got {window.shape[0]}")

    price = window[:, PRICE_COLS]
    price_feat = _log_returns(price)

    volume = window[:, VOLUME_COL:VOLUME_COL + 1]
    volume = tf.math.log1p(volume)
    vol_mean = tf.reduce_mean(volume, axis=0, keepdims=True)
    vol_std = tf.math.reduce_std(volume, axis=0, keepdims=True)
    volume = tf.math.divide_no_nan(volume - vol_mean, vol_std + EPS)

    x_feat = tf.concat([price_feat, volume], axis=1)
    last_close = window[-1, TARGET_COL]
    return x_feat, last_close


def predict_prices_from_window(window, model=None, model_path=None, return_numpy=True):
    x_feat, last_close = _features_from_window(window)
    if model is None:
        if model_path is None:
            model_path = SAVE_FILE + "_best.keras"
        model = keras.models.load_model(model_path)
    pred_log_return = model(tf.expand_dims(x_feat, axis=0), training=False)
    pred_log_return = tf.squeeze(pred_log_return, axis=0)
    pred_prices = last_close * tf.exp(pred_log_return)
    if return_numpy:
        return pred_prices.numpy()
    return pred_prices


def main():
    train_ds = get_dataset(TRAIN_FILES, BATCH_SIZE, shuffle=True)
    val_ds = get_dataset(VAL_FILES, BATCH_SIZE, shuffle=False)

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.Huber(),
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.MeanSquaredError(name="mse"),
        ],
    )

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            SAVE_FILE + "_best.keras", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
    ]

    model.fit(
        x=train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(SAVE_FILE + ".keras")


if __name__ == "__main__":
    main()
