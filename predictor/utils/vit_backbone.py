import keras
from keras import layers

@keras.saving.register_keras_serializable(package="Custom")
class ViTBackbone(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # তোমার existing init code এখানে থাকবে

    def call(self, inputs, training=False):
        # তোমার existing forward logic
        return inputs  # example

    def get_config(self):
        config = super().get_config()
        # যদি extra params থাকে, এখানে add করো
        return config
