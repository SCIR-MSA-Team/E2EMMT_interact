# E2EMMT
End2End Multimodal Transformer


## Data Processing
 
python pre_process.py

## Run

### CLI
python run.py -b 8 --lr 0.0001  --n-epochs 40 --exp-dir model_dir/mosei_imagenet_8_0.0001_40_fusion --imagenet_pretrain True --fstride 16 --tstride 16 -w 12 

python run.py --batch-size 8 --time_dim_split True --face_size 128 --v_patch_num 256 --a_patch_num 256 --exp-dir model_dir/temp_imagenet_8_0.0001_40_fusion --lr 1e-4  --n-epochs 40  --a_imagenet_pretrain True --v_imagenet_pretrain True  --fstride 16 --tstride 16 -w 12 --dataset iemocap --seed 1234

### SBATCH
See generate_slurms.py

## Results
See https://docs.qq.com/sheet/DTEFpbFZMWmZESUR6?tab=BB08J2

## 消融实验
- 去掉interact部分代码，只用3个cls predict的结果+3个cls的cat predict结果
- 调研某一层的重要性：
    - 从最后一层倒着向前选择1层，2层，3层看实验结果
    - 在12层中去掉第1层，第2层，，，看实验结果
- TA，TV，AV实验结果
- interact去掉attention部分