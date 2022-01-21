python main.py \
    kinetics RGB --arch resnet152 \
    --num_segments 8 --gpus 0 1 2 3  --gd 60\
    --lr 0.0002 --lr_steps 20 30 --epochs 40 \
    --batch-size 26  -j 8 --dropout 0.5 \
    --consensus_type=avg --eval-freq=1 \
    --npb --wd 1e-4\
    --TCP  --lr_x_factor 50\
    --store_name K400_TCP_TSN_16f_R152 \
    --load_GCP_from  pretrained_models/ImgeNet11K_1K_preact_R152_GCP.pth.tar
