model=So_vit_14
./distributed_train.sh 8 \
/imagenetpytorch \
--model $model \
-b 64 \
--lr 1e-3 \
--weight-decay .05 \
--img-size 224 \
--amp 