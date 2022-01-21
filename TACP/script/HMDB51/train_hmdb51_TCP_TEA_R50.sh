python main.py \
    hmdb51 RGB --arch tea50 \
    --num_segments 8 --gpus 0 1 --gd 20 \
    --lr 0.001 --lr_steps 5 10 --epochs 15 \
    --batch-size 32  -j 8 --dropout 0.5 \
    --consensus_type=avg --eval-freq=1 \
    --shift --shift_div=8 --shift_place=blockres \
    --TCP \
    --store_name  HMDB51_TCP_TEA_R50_8f \
    --load_TCP_from  K400_TCP_TEA_R50_8f.pth.tar
