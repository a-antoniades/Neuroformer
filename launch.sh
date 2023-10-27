CUDA_VISIBLE_DEVICES=5 python neuroformer_Visnav_NF_1.5.py --dataset medial --seed 69
CUDA_VISIBLE_DEVICES=6 python neuroformer_Visnav_NF_1.5.py --dataset medial --seed 25
CUDA_VISIBLE_DEVICES=7 python neuroformer_Visnav_NF_1.5.py --dataset medial --seed 420


CUDA_VISIBLE_DEVICES=2 python neuroformer_Visnav_NF_1.5.py --dataset medial --seed 69
CUDA_VISIBLE_DEVICES=3 python neuroformer_Visnav_NF_1.5.py --dataset lateral --seed 25
CUDA_VISIBLE_DEVICES=4 python neuroformer_Visnav_NF_1.5.py --dataset lateral --seed 420

export CUDA_VISIBLE_DEVICES=7 
python neuroformer_Visnav_NF_1.5.py \
       --dataset medial \
       --seed 69 \
       --config ./configs/NF_1.5/VisNav_VR_Expt/mlp_only/mconf.yaml

export CUDA_VISIBLE_DEVICES=6 
python neuroformer_Visnav_NF_1.5.py \
       --dataset medial \
       --seed 25 \
       --config ./configs/NF_1.5/VisNav_VR_Expt/gru2_only_cls/mconf.yaml

export CUDA_VISIBLE_DEVICES=6 
python neuroformer_Visnav_NF_1.5.py \
       --dataset lateral \
       --seed 25 \
       --config ./configs/NF_1.5/VisNav_VR_Expt/gru2_only_cls/mconf.yaml

export CUDA_VISIBLE_DEVICES=5 
python neuroformer_Visnav_NF_1.5.py \
       --dataset medial \
       --seed 69 \
       --config ./configs/NF_1.5/VisNav_VR_Expt/gru2_only/mconf.yaml



    



