# E2EMMT
End2End Multimodal Transformer


## Data Processing
 
python pre_process.py

## Run

### CLI
python run.py -b 8 --lr 0.0001  --n-epochs 40 --exp-dir model_dir/mosei_imagenet_8_0.0001_40_fusion --imagenet_pretrain True --fstride 16 --tstride 16 -w 12 

### SBATCH
See generate_slurms.py

## Results
See https://docs.qq.com/sheet/DV0phRlV2ek9EeXNL?tab=BB08J2  and https://docs.qq.com/sheet/DV0ZrZlpEeVBabnB3?tab=BB08J2


