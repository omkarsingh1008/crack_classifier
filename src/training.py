from src.utils.common_utils import read_config
from src.utils.model import model
from src.utils.data_mngmt import data_prepare
import argparse
import os
import tensorflow as tf
from src.utils.common_utils import evaluate_model
def train(config_path):
    content=read_config(config_path)
    loss=content['params']['loss_function']
    metrics=content['params']['metrics']
    optimizer=content['params']['optimizer']
    valid_size=content['params']['validation_datasize']
    test_size=content['params']['test_datasize']
    train_size=content['params']['train_datasize']
    epoch=content['params']['epochs']
    batch_size=content['params']['batch_size']
    num_classes=content['params']['num_classes']
    path_ckpt=os.path.join(content['artifacts']["artifacts_dir"],content['artifacts']["checkpoint_dir"],"cp.ckpt")
    train_Data,test_Data,valid_Data=data_prepare(valid_size,test_size,train_size)
    models=model(optimizer,loss,metrics,num_classes)
    #checkpoint_path = "model_checkpoint/cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path_ckpt,
                                                 save_best_only=True, save_weights_only=True, verbose=1) 
    history=models.fit(train_Data,validation_data=valid_Data,epochs=epoch,callbacks=[
    tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True)
    ,cp_callback])
    evaluate_model(models,test_Data,config_path,history)
    models.save(os.path.join(content['artifacts']['artifacts_dir'],content['artifacts']['model_dir']+"model.h5"))
    