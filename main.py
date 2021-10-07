from src.training import train
import argparse

if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--config',default="config.yaml")
    args.add_argument('--train',help="0 for training",default=1)
    parsed_args=args.parse_args()
  
    if int(parsed_args.train)==0:
    
        config_path=parsed_args.config
        train(config_path)

