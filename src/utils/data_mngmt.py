from src.utils.common_utils import dataframe
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
def data_prepare(valid_size,test_size,train_size):
    data_def = dataframe()
    total_len=len(data_def)
    train_size=int(total_len*(train_size/100))
    valid_size=int(total_len*(valid_size/100))
    test_size=int(total_len*(test_size/100))
    train_df=data_def[:train_size]
    size=train_size+valid_size
    valid_df=data_def[train_size:size]
    test_df=data_def[size:]
    train_df.to_csv("train.csv",index=False)
    valid_df.to_csv("valid.csv",index=False)
    test_df.to_csv("test.csv",index=False)
    train_gen=ImageDataGenerator(rescale=1./255.,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            vertical_flip=True,
                            shear_range=0.2)
    valid_gen=ImageDataGenerator(rescale=1./255.)
    test_gen=ImageDataGenerator(rescale=1./255.)
    train_Data=train_gen.flow_from_dataframe(train_df,x_col='filename',y_col='labels',
                                            target_size=(150,150),batch_size=32,class_mode="binary")
    valid_Data=valid_gen.flow_from_dataframe(valid_df,x_col='filename',y_col='labels',
                                            target_size=(150,150),batch_size=32,class_mode="binary")
    test_Data=test_gen.flow_from_dataframe(test_df,x_col='filename',y_col='labels',
                                            target_size=(150,150),batch_size=32,class_mode="binary")
    return train_Data,test_Data,valid_Data

