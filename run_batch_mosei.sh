#!/bin/bash
#SBATCH -J mosei                              # 作业名为 test
#SBATCH -o layer_0.33_mosei_lr_1e-4_batch_size_8_1234.out                           # 屏幕上的输出文件重定向到 test.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH --gres=gpu:tesla_v100s-pcie-32gb:1
#SBATCH -t 24:00:00                            # 任务运行的最长时间为 24 小时

source ~/.bashrc
source /users5/ywu/Env/MMT/bin/activate

python run.py --batch-size 8 --time_dim_split True --face_size 128  --exp-dir model_dir/mosei_lr_1e-4_batch_size_8_imagenet_8_0.0001_40_fusion_2_12_1234 --lr 1e-4  --n-epochs 40  --a_imagenet_pretrain True --v_imagenet_pretrain True  --fstride 16 --tstride 16 -w 12 --dataset mosei --seed 1234 
