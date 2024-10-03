model='DiT-g/2'
params=" \
            --qk-norm \
            --model ${model} \
            --rope-img base512 \
            --rope-real \
            "
deepspeed --include localhost:1,2 --master_port 25000 hydit/train_deepspeed_relight.py ${params}  "$@"