model=So_vit_19
./distributed_train.sh 8 \
/data/datasets/ImageNet/images \
--model $model \
-b 64 \
--lr 1e-3 \
--weight-decay .05 \
--img-size 224 \
--amp 