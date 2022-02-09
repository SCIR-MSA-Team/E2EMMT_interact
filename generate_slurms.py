import os
import random


def output_slurm(dataset, pretrained_type, bz, lr, epoch, scale_value, slurm_root_path):
    gpu_device = random.choice([ 'tesla_v100-sxm2-16gb', 'tesla_v100s-pcie-32gb'])

    job_name = f'{dataset[:2]}_{pretrained_type[:2]}_{bz}_{lr}_{epoch}_{scale_value}'
    all_name = f'{dataset}_{pretrained_type}_{bz}_{lr}_{epoch}_{scale_value}'
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
            "#SBATCH -t 12:00:00",
            f"#SBATCH --gres=gpu:{gpu_device}:{gpu_num}",
            "",
            "source ~/.bashrc",
            "",
            "source activate gpu12py38",
            "",
            "source  /users/ywu/E2EMT/ast-master/venvast/bin/activate",
            "",
            f"python run.py --dataset {dataset} -b {bz} --lr {lr}  --scale {scale_value}  --n-epochs {epoch} --exp-dir {model_path}{all_name} --fstride 16 --tstride 16 -w 12"
        ]
    elif pretrained_type == "scratch":
        slurm_template = [
        "#!/bin/bash",
        f"#SBATCH -J {job_name}                         ",
        f"#SBATCH -o {output_file}                           ",
        "#SBATCH -p compute                            ",
        "#SBATCH -N 1                                  ",
        "#SBATCH -t 12:00:00",
        f"#SBATCH --gres=gpu:{gpu_device}:{gpu_num}",
        "",
        "source ~/.bashrc",
        "",
        "source activate gpu12py38",
        "",
        "source  /users/ywu/E2EMT/ast-master/venvast/bin/activate",
        "",
        f"python run.py --dataset {dataset} -b {bz} --lr {lr} --n-epochs {epoch} --scale {scale_value} --exp-dir {model_path}{all_name} --imagenet_pretrain False --fstride 16 --tstride 16 -w 12"
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
    slurm_root_path = 'slurm_sparse_transform_face128_1230/'
    bz = 8
    epoch = 40

    # pretrained_type = 'imagenet'

    for dataset in ["mosei", "iemocap"]:
        for scale_value in [1/(256*256*3)]:
            for lr in [0.0001]:
                for pretrained_type in ['imagenet', 'scratch']:
                    slurm_files.append('sbatch ' + output_slurm(dataset, pretrained_type, bz, lr, epoch, scale_value, slurm_root_path))

    with open(f'{slurm_root_path}run_slurms_1230.sh', 'w') as f:
        f.write('\n'.join(slurm_files))
