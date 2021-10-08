import tensorflow as tf
from src.utils.common_utils import read_config
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
def predict(config_path,img_path):
    content=read_config(config_path)
    model_path=os.path.join(content['artifacts']['artifacts_dir'],content['artifacts']["model_dir"])
    model=tf.keras.models.load_model(model_path+"/"+"model.h5")
    test_image = image.load_img(img_path, target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    if result[0][0]<=0.5:
        print("negative")
    else:
        print("positive")