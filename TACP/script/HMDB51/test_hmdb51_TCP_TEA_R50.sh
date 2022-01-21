python test_models.py \
    hmdb51 --arch tea50 \
    --test_crops 3 --ten_clips_sample \
    --test_segments 8 \
    --full_res --input_size 256 \
    --batch_size 2 \
    --softmax \
    --weights results/HMDB51_TCP_TEA_R50_8f/model_best.pth.tar \
    --TCP
