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
