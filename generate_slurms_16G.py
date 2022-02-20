import os
import random


def output_slurm(dataset, pretrained_type, bz, lr, epoch, seed, face_size, layer_loss_factor, constrastive_loss_factor, slurm_root_path):
    gpu_device = random.choice(['tesla_v100-pcie-32gb', 'tesla_v100s-pcie-32gb'])
    gpu_node = "gpu24"
    
    job_name = f'{dataset}'
    all_name = f'{dataset}_{pretrained_type}_{bz}_{lr}_{epoch}_{seed}_{face_size}_{layer_loss_factor}_{constrastive_loss_factor}'
    output_path = f'{slurm_root_path}slurm_outs/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = f'{output_path}{all_name}.out'
    
    model_path = f'{slurm_root_path}model_dir/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)  

    gpu_num = 1  

    if pretrained_type == "imagenet":
        slurm_template = [
            "#!/bin/bash",
            f"#SBATCH -J {job_name}                               ",
            f"#SBATCH -o {output_file}                            ",
            "#SBATCH -p compute                           ",
            "#SBATCH -N 1                                  ",
            "#SBATCH -t 6:00:00",
            f"#SBATCH -w {gpu_node}",
            f"#SBATCH --gres=gpu:{gpu_device}:{gpu_num}",
            "",
            "source ~/.bashrc",
            "",
            "source /users5/ywu/Env/MMT/bin/activate",
            "",
            f"python run.py --batch-size {bz} --early_stop 3 --time_dim_split True  --constrastive_loss_factor {constrastive_loss_factor} --layer_loss_factor {layer_loss_factor}  --face_size {face_size}  --exp-dir {model_path}{all_name}  --lr {lr} --n-epochs {epoch}  --a_imagenet_pretrain True --v_imagenet_pretrain True  --fstride 16 --tstride 16 -w 12 --dataset {dataset} --seed {seed}"
        ]
    elif pretrained_type == "scratch":
        slurm_template = [
            "#!/bin/bash",
            f"#SBATCH -J {job_name}                               ",
            f"#SBATCH -o {output_file}                            ",
            "#SBATCH -p compute                           ",
            "#SBATCH -N 1                                  ",
            "#SBATCH -t 6:00:00",
            f"#SBATCH -w {gpu_node}",
            f"#SBATCH --gres=gpu:{gpu_device}:{gpu_num}",
            "",
            "source ~/.bashrc",
            "",
            "source /users5/ywu/Env/MMT/bin/activate",
            "",
            f"python run.py --batch-size {bz} --early_stop 3 --time_dim_split True --constrastive_loss_factor {constrastive_loss_factor} --layer_loss_factor {layer_loss_factor}  --face_size {face_size}  --exp-dir {model_path}{all_name}  --lr {lr} --n-epochs {epoch}  --a_imagenet_pretrain False --v_imagenet_pretrain False  --fstride 16 --tstride 16 -w 12 --dataset {dataset} --seed {seed}"
        ]

    slurm_path = f'{slurm_root_path}slurms/'
    if not os.path.exists(slurm_path):
        os.makedirs(slurm_path)
    slurm_file = f'{slurm_path}{all_name}.slurm'

    with open(slurm_file, 'w') as f:
        f.write('\n'.join(slurm_template))

    return slurm_file


if __name__ == '__main__':
    slurm_files = []
    slurm_root_path = 'push_pull_iter_loss_progressive_0219/'
    bz = 8
    epoch = 40
    pretrained_type = 'imagenet'
    lr = 1e-4

    for dataset in ["mosei", "iemocap"]:
        for layer_loss_factor in [0, 1, 10, 50]:
            for constrastive_loss_factor in [0, 0.01, 0.25]:
                for seed in [1235]:
                    for face_size in [64, 128]:
                        slurm_files.append('sbatch ' + output_slurm(dataset, pretrained_type, bz, lr, epoch, seed, face_size, layer_loss_factor, constrastive_loss_factor, slurm_root_path))

    with open(f'{slurm_root_path}run_slurms_0219.sh', 'w') as f:
        f.write('\n'.join(slurm_files))

# python run.py --batch-size 8 --early_stop 5 --time_dim_split True  --constrastive_loss_factor 0.25 --layer_loss_factor 0.1  --face_size 128  --exp-dir pull_push_iter_contrastive_loss_progressive_search_0218/model_dir/iemocap_imagenet_8_0.0001_40_1234_128_0.1_0.25 --lr 1e-4 --n-epochs 0  --a_imagenet_pretrain True --v_imagenet_pretrain True  --fstride 16 --tstride 16 -w 12 --dataset iemocap --seed 1234