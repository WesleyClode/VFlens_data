from data import DataSetLoader
import argparse
import numpy as np
import time
import os

def train(args):
    print(args)
    dataset = DataSetLoader(args.data_name, test_ratio=args.test_ratio, pred_ratio=args.pred_ratio)


def config():
    parser = argparse.ArgumentParser(description='PGMC')
    parser.add_argument('--seed', default=125, type=int) #123
    parser.add_argument('--save_dir', type=str, help='The saving directory')
    parser.add_argument('--save_id', type=int, help='The saving log id')
    parser.add_argument('--pred_ratio', type=float, default=0.5) 
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--data_name', default='credit', type=str,
                        help='The dataset name: credit,...')
    
    args = parser.parse_args()

    ### configure save_fir to save all the info

    now = int(round(time.time()*1000))
    now02 = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now/1000))
    if args.save_dir is None:
        args.save_dir = args.data_name+"_" + ''.join(now02)
    if args.save_id is None:
        args.save_id = np.random.randint(20)
    args.save_dir = os.path.join("log", args.save_dir)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    return args

if __name__ == '__main__':
    args = config()
    train(args)