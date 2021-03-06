import segmentation_models as sm


class Config:
    BACKBONE = 'efficientnetb3'
    BATCH_SIZE = 1

    preprocess_input = sm.get_preprocessing(BACKBONE)
    activation = 'softmax'

    epochs = 50
    threshold = 0.5
    class_names = []
