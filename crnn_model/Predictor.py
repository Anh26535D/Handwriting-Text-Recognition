import tensorflow as tf
import keras.backend as K
import numpy as np
from .vocab import Vocab
from .Model import build_model
from .utils import preprocess_image

class Predictor():

    def __init__(self, weight_path, max_len, image_width, image_height) -> None:
        self.weight_path = weight_path
        self.max_len = max_len
        self.image_width = image_width
        self.image_height = image_height

        model = build_model(image_width=self.image_width, image_height=self.image_height)
        model.load_weights(self.weight_path)
        self.prediction_model = tf.keras.models.Model(model.get_layer(
            name="image").input, model.get_layer(name="dense2").output)

    def predict(self, image_path):
        image = preprocess_image(image_path, img_size=(self.image_width, self.image_height))
        image = tf.reshape(image, (-1, self.image_width, self.image_height, 1))
        pred = self.prediction_model.predict(image, verbose=0)
        # input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # res = K.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :self.max_len]
        # res = tf.gather(res[0], tf.where(tf.math.not_equal(res, -1)))
        # res = tf.strings.reduce_join(Vocab.num_to_char(res)).numpy().decode("utf-8")
        pred_texts = self.decode_batch_predictions(pred)
        return pred_texts[0]
    
    
    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]

        results = K.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
            :, :self.max_len
        ]
        # Iterate over the results and get back the text.
        output_text = []
        for res in results:
            res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
            res = tf.strings.reduce_join(
                Vocab.num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text
