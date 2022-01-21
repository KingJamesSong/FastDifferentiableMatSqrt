model=So_vit_7
./distributed_train.sh 8 \
/imagenetpytorch \
--model $model \
-b 64 \
--lr 1e-3 \
--weight-decay .03 \
--img-size 224 \
--amp \
--resume