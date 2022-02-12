#!/bin/bash
#SBATCH -J test                               # 作业名为 test
#SBATCH -o test.out                           # 屏幕上的输出文件重定向到 test.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH --ntasks-per-node=3                   # 单节点启动的进程数为 2
#SBATCH --cpus-per-task=3                     # 单任务使用的 CPU 核心数为 4
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH -t 100:00:00                            # 任务运行的最长时间为 100 小时

source ~/.bashrc

# 设置运行环境
conda activate SparseEnd2End
cd /users10/zyzhang/multimodel/E2EMMT
# 输入要执行的命令，例如 ./hello 或 python test.py 等

python run.py --time_dim_split True --face_size 128 --v_patch_num 256 --a_patch_num 256 --exp-dir model_dir/mosei_imagenet_8_0.0001_40_fusion -b 8 --lr 1e-4  --n-epochs 40  --a_imagenet_pretrain True --v_imagenet_pretrain True  --fstride 16 --tstride 16 -w 12 --dataset mosei > output_mosei.txt 2>&1
