from src.training import train
from src.predict import predict
import argparse

if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--config',default="config.yaml")
    args.add_argument('--train',help="0 for training",default=1)
    args.add_argument('--test',help="0 for testing",default=1)
    args.add_argument('--image',help="image path")
    parsed_args=args.parse_args()
  
    if int(parsed_args.train)==0:
    
        config_path=parsed_args.config
        train(config_path)
    if int(parsed_args.test)==0:
        config_path=parsed_args.config
        image_path=parsed_args.image
        predict(config_path,image_path)

