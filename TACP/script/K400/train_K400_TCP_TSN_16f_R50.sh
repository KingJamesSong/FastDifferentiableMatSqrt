python main.py \
    kinetics RGB --arch resnet50 \
    --num_segments 8 --gpus 0 1 2 3  --gd 20\
    --lr 0.00375 --lr_steps 20 40 --epochs 50 \
    --batch-size 96  -j 8 --dropout 0.5 \
    --consensus_type=avg --eval-freq=1 \
    --npb --wd 1e-4\
    --TCP --lr_x_factor 5\
    --store_name K400_TCP_TSN_16f_R50 \
    --load_GCP_from  pretrained_models/ImgeNet1K_GCP_R50.pth.tar
