python main.py \
    kinetics RGB --arch tea50 \
    --num_segments 8 --gpus 0 1 2 3 --gd 20\
    --lr 0.01 --lr_steps 20 30 --epochs 40 \
    --batch-size 32  -j 8 --dropout 0.5 \
    --consensus_type=avg --eval-freq=1 \
    --shift --shift_div=8 --shift_place=blockres \
    --npb --wd 1e-4\
    --TCP --lr_x_factor 10\
    --store_name K400_TCP_TEA_8f_R50 \
    --load_GCP_from pretrained_models/ImageNet1K_Res2Net50_GCP_R50.pth.tar
