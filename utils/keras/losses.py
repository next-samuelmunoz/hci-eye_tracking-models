

from tensorflow.contrib.keras import backend as K


def euclidean(y_true, y_pred):
    return K.mean(
        K.sqrt(
            K.sum(
                K.square(
                    y_true - y_pred
                ),
                axis=1
            )
        )
    )
