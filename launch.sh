# CUDA_VISIBLE_DEVICES=5 python neuroformer_Visnav_NF_1.5.py --dataset medial --seed 69
# CUDA_VISIBLE_DEVICES=6 python neuroformer_Visnav_NF_1.5.py --dataset medial --seed 25
# CUDA_VISIBLE_DEVICES=7 python neuroformer_Visnav_NF_1.5.py --dataset medial --seed 420


# CUDA_VISIBLE_DEVICES=2 python neuroformer_Visnav_NF_1.5.py --dataset medial --seed 69
# CUDA_VISIBLE_DEVICES=3 python neuroformer_Visnav_NF_1.5.py --dataset lateral --seed 25
# CUDA_VISIBLE_DEVICES=4 python neuroformer_Visnav_NF_1.5.py --dataset lateral --seed 420

# export CUDA_VISIBLE_DEVICES=7 
# python neuroformer_Visnav_NF_1.5.py \
#        --dataset medial \
#        --seed 69 \
#        --config ./configs/NF_1.5/VisNav_VR_Expt/mlp_only/mconf.yaml

# export CUDA_VISIBLE_DEVICES=6 
# python neuroformer_Visnav_NF_1.5.py \
#        --dataset medial \
#        --seed 25 \
#        --config ./configs/NF_1.5/VisNav_VR_Expt/gru2_only_cls/mconf.yaml

# export CUDA_VISIBLE_DEVICES=6 
# python neuroformer_Visnav_NF_1.5.py \
#        --dataset lateral \
#        --seed 25 \
#        --config ./configs/NF_1.5/VisNav_VR_Expt/gru2_only_cls/mconf.yaml

# export CUDA_VISIBLE_DEVICES=5 
# python neuroformer_Visnav_NF_1.5.py \
#        --dataset medial \
#        --seed 69 \
#        --config ./configs/NF_1.5/VisNav_VR_Expt/gru2_only/mconf.yaml

# CUDA_VISIBLE_DEVICES=3 \
# python neuroformer_train.py \
#        --dataset visnav_tigre \
#        --seed 420 \
#        --wandb false \
#        --title debug \
#        --config ./configs/Visnav/tigre/mconf_predict_depth.yaml

# CUDA_VISIBLE_DEVICES=6 \
# python neuroformer_train.py \
#        --dataset visnav_tigre \
#        --seed 420 \
#        --wandb true \
#        --title pretrain_depth_luminance \
#        --config ./configs/Visnav/tigre/mconf_predict_depth_luminance.yaml

CUDA_VISIBLE_DEVICES=5 \
python neuroformer_train.py \
       --dataset visnav_tigre \
       --seed 25 \
       --wandb true \
       --title pretrain_finetune_depth \
       --config ./configs/Visnav/tigre/mconf_pretrain.yaml \
       --resume "./models/NF.15/Visnav_VR_Expt/visnav_tigre/Neuroformer/pretrain/(state_history=6,_state=6,_stimulus=0,_behavior=0,_self_att=6,_modalities=(n_behavior=25))/25/model.pt" \
       --loss_bprop depth


    

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export WORLD_SIZE=4  # Adjust this to the number of GPUs you want to use
# export OMP_NUM_THREADS=4  # Adjust this number based on your CPU cores and workload
# export MASTER_PORT=29508  # Change this to an available port

# torchrun --nnodes=1 --nproc_per_node=$WORLD_SIZE --rdzv_backend=c10d \
# --rdzv_endpoint=localhost:$MASTER_PORT neuroformer_train.py \
#     --dataset visnav_tigre \
#     --seed 420 \
#     --wandb true \
#     --dist \
#     --title pretrain_predict_dl_12l_1024e \
#     --config ./configs/Visnav/tigre/mconf_predict_depth_luminance.yaml


