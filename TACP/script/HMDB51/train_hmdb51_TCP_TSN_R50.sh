python main.py \
    hmdb51 RGB --arch resnet50 \
    --num_segments 8 --gpus 0 1 \
    --gd 20 --lr 0.001 --lr_steps 5 10 \
    --epochs 15 --batch-size 52 -j 8 \
    --dropout 0.8 --consensus_type=avg --eval-freq=1 \
    --wd 5e-4 --TCP --store_name HMDB51_TCP_TSN_R50_8f \
    --lr_x_factor 5 --load_TCP_from  K400_TCP_TSN_R50_8f.pth.tar
