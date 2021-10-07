import os
import pandas as pd
import yaml
import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
import numpy as np

def read_config(config_path):
    print("hello")
    with open(config_path) as config:
        content = yaml.safe_load(config)
        
        
        return content
#read_config("/media/omkar/omkar/computer_vision/program/crack_classifier/crack/crack_classifier/config.yaml")
def dataframe():
    data_df=pd.DataFrame(columns=['filename','labels'])
    path="dataset/"
    image_path=[]
    labels=[]
    for i in os.listdir(path):
        label=i
        for j in os.listdir(path+i):
            img_path=path+i+"/"+j
            image_path.append(img_path)
            labels.append(label)
    data_df['filename']=image_path
    data_df['labels']=labels
    data_df=data_df.sample(frac=1)
    return data_df

def evaluate_model(model,test_data,config_path,history):
    content=read_config(config_path)
    path=os.path.join(content['logs']["logs_dir"],content['logs']["general_logs"])
    result=model.evaluate(test_data,verbose=0)
    loss=result[0]
    accuracy=result[1]
    
    print("test loss: {:.5f}".format(loss))
    print("test acc: {:.2f}".format(accuracy*100))
    
    y_pred=np.squeeze((model.predict(test_data)>=0.5).astype(np.int))
    cm=confusion_matrix(test_data.labels,y_pred)
    clr = classification_report(test_data.labels, y_pred, target_names=["NEGATIVE", "POSITIVE"])
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
    plt.xticks(ticks=np.arange(2) + 0.5, labels=["NEGATIVE", "POSITIVE"])
    plt.yticks(ticks=np.arange(2) + 0.5, labels=["NEGATIVE", "POSITIVE"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(path,"confusion_matrix.jpg"))
    
    
    print("Classification Report:\n----------------------\n", clr)
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.savefig(os.path.join(path,"plot.jpg"))
    