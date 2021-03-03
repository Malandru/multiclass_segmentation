import segmentation_models as sm
import numpy as np
import visualize
from unet.config import Config
from tensorflow import keras


class Inference:
    def __init__(self, config=None):
        if config is None:
            config = Config()  # Use default configuration
        self.model = sm.Unet(config.BACKBONE, classes=len(config.class_names), activation=config.activation)

        self.x_train = np.array([])  # numpy array with shape (X, H, W, 3) ==> X images with 3 color channels
        self.y_train = np.array([])  # numpy array with shape (Y, H, W, 1) ==> Y images with mask result
        self.config = config

    def create_model(self):
        optimizer = keras.optimizers.Adam(self.config.LR)
        class_weights = np.random.uniform(1, 100, len(self.config.class_names))

        dice_loss = sm.losses.DiceLoss(class_weights=class_weights)
        focal_loss = sm.losses.CategoricalFocalLoss()
        total_loss = dice_loss + (2 * focal_loss)

        threshold = self.config.threshold
        metrics = [sm.metrics.IOUScore(threshold=threshold), sm.metrics.FScore(threshold=threshold)]
        self.model.compile(optimizer, total_loss, metrics)

    def load_model(self, model_path=''):
        if len(model_path) == 0:
            self.model = keras.models.load_model(model_path)

    def save_model(self, filename):
        self.model.save(filename)

    def set_train_params(self, x, y):
        self.x_train = x
        self.y_train = y

    def train(self):
        self.model.fit(x=self.x_train, y=self.y_train, batch_size=self.config.BATCH_SIZE, epochs=self.config.epochs)

    def test_predictions(self, x_test, y_test):
        predictions = self.model.predict(x_test)
        random_index = np.random.randint(0, len(x_test))

        test_input = x_test[random_index]
        expected_output = y_test[random_index]
        prediction_output = predictions[random_index]

        visualize.prepare_figure(
            test_input=test_input,
            expected_output=expected_output,
            prediction_output=prediction_output
        )
