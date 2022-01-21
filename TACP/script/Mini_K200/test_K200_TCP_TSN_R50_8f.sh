python test_models.py \ 
    mini_kinetics_200 --arch resnet50 \
    --test_crops 3 --ten_clips_sample \
    --test_segments 8 \
    --full_res --input_size 256 \
    --batch_size 2 \
    --softmax \
    --weights results/K200_TCP_TSN_8f_R50/model_best.pth.tar \
    --TCP
