task_flag="dit_g2_full_1024p_relight"                         # the task flag is used to identify folders.
resume_module_root=/home/zhaoyi/media/vllm_ckpt/HunyuanDiT-v1.1/t2i/model/pytorch_model_module.pt # checkpoint root for model resume
index_file="/home/zhaoyi/media/dataset/ffhq/1024relighted/ffhq_relight/jsons/ffhq_relighted.json"                                             # index file for dataloader
results_dir=./log_EXP                                         # save root for results
batch_size=1                                                  # training batch size
image_size=1536                                               # training image resolution
grad_accu_steps=1                                             # gradient accumulation
warmup_num_steps=0                                            # warm-up steps
lr=0.0001                                                     # learning rate
ckpt_every=9999999                                            # create a ckpt every a few steps.
ckpt_latest_every=9999999                                     # create a ckpt named `latest.pt` every a few steps.
ckpt_every_n_epoch=2                                          # create a ckpt every a few epochs.
epochs=8                                                      # total training epochs
relight_mode="fg"

sh $(dirname "$0")/run_g_relight.sh \
    --task-flag ${task_flag} \
    --noise-schedule scaled_linear --beta-start 0.00085 --beta-end 0.018 \
    --predict-type v_prediction \
    --uncond-p 0 \
    --uncond-p-t5 0 \
    --index-file ${index_file} \
    --random-flip \
    --lr ${lr} \
    --batch-size ${batch_size} \
    --image-size ${image_size} \
    --global-seed 999 \
    --grad-accu-steps ${grad_accu_steps} \
    --warmup-num-steps ${warmup_num_steps} \
    --use-flash-attn \
    --use-fp16 \
    --extra-fp16 \
    --results-dir ${results_dir} \
    --resume \
    --resume-module-root ${resume_module_root} \
    --epochs ${epochs} \
    --ckpt-every ${ckpt_every} \
    --ckpt-latest-every ${ckpt_latest_every} \
    --ckpt-every-n-epoch ${ckpt_every_n_epoch} \
    --log-every 10 \
    --deepspeed \
    --use-zero-stage 2 \
    --gradient-checkpointing \
    --use-style-cond \
    --size-cond 1024 1024 \
    --relight_mode ${relight_mode} \
    "$@"
