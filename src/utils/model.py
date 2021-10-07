import tensorflow as tf

def incep_block(inputs):
    conv1=tf.keras.layers.Conv2D(64,(1,1),activation='relu',padding="same")(inputs)
    
    conv1_3reduce=tf.keras.layers.Conv2D(96,(1,1),activation='relu',padding="same")(inputs)
    conv3_3reduce=tf.keras.layers.Conv2D(128,(3,3),padding='same',activation='relu')(conv1_3reduce)
    
    conv3=tf.keras.layers.Conv2D(128,(3,3),padding='same',activation='relu')(inputs)
    
    conv1_5reduce=tf.keras.layers.Conv2D(16,(1,1),padding='same',activation='relu')(inputs)
    conv5_5reduce=tf.keras.layers.Conv2D(32,(5,5),padding='same',activation='relu')(conv1_5reduce)
    
    conv5=tf.keras.layers.Conv2D(32,(5,5),padding='same',activation='relu')(inputs)
    
    #maxpl1_reduce=tf.keras.layers.MaxPool2D((3,3),padding='same')(inputs)
    #maxpl1_conv=tf.keras.layers.Conv2D(32,(1,1),padding='same',activation='relu')(maxpl1_reduce)
    
    contcat=tf.keras.layers.concatenate([conv1,conv3_3reduce,conv3,conv5_5reduce,conv5],axis=3, name=None)
    return contcat

def model(optimizer,loss,metrics,num_classes,image_shape=(15,15,3)):
    input=tf.keras.layers.Input(shape=image_shape)
    x=tf.keras.layers.Conv2D(16,3,activation="relu")(input)
    x=tf.keras.layers.MaxPool2D((2,2))(x)

    x=tf.keras.layers.Conv2D(32,3,activation="relu")(x)
    x=tf.keras.layers.MaxPool2D((2,2))(x)
    x=incep_block(x)
    x=tf.keras.layers.GlobalAveragePooling2D()(x)

    output=tf.keras.layers.Dense(num_classes,activation="sigmoid")(x)

    model=tf.keras.Model(input,output)
    tf.keras.utils.plot_model(
    model, to_file='artifacts/plots/model.png',show_shapes=True
    )

    model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
    return model