import argparse
import pandas as pd
import numpy as np

def cal(values):
    mean = round(np.mean(values)*100, 2)
    std = round(np.std(values)*100, 2)
    return str(mean)+'±'+str(std)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True,type=str)
    parser.add_argument('--lr',required=True,type=str)
    parser.add_argument('--batch_size',required=True,type=int)
    # parser.add_argument('--tds',required=True,type=str)
    args=parser.parse_args()
    print('args',args)
    print('args.dataset',args.dataset)
    data=pd.DataFrame(columns=['seed','lr','batch_size','acc','f1'])
    save_path='./extractDataFile/{}_lr_{}_batch_size_{}.csv'.format(args.dataset, args.lr, args.batch_size)

    for seed in range(1234,1239):
        result_path='./model_dir/{}_lr_{}_batch_size_{}_imagenet_8_0.0001_40_fusion_2_12_{}/eval_result.csv'.format(args.dataset, args.lr, args.batch_size,seed)
        print('result_path',result_path)
        temp=[seed, args.lr, args.batch_size]
        read=pd.read_csv(result_path)
        temp.append(float(list(read.columns)[0]))
        print(read)
        read.columns=['eva']
        temp.append(read.loc[2,'eva'])
        print(read)
        print(temp)
        data.loc[len(data)]=temp
    
    data.to_csv(save_path,index=False)

