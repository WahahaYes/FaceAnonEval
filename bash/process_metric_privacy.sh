BATCH_SIZE=16

for k in 4 6 8
do

    for eps in 0.1 0.3 0.5 1.0 2.0 3.0 5.0 7.0 10.0 15.0
    do
        python process_dataset.py --dataset CelebA_test --privacy_mechanism metric_privacy --batch_size $BATCH_SIZE --metric_privacy_k $k --dp_epsilon $eps
        python process_dataset.py --dataset lfw --privacy_mechanism metric_privacy --batch_size $BATCH_SIZE --metric_privacy_k $k --dp_epsilon $eps

        python evaluate_mechanism.py --dataset CelebA_test --privacy_mechanism metric_privacy --batch_size $BATCH_SIZE --metric_privacy_k $k --dp_epsilon $eps --evaluation_method rank_k 
        python evaluate_mechanism.py --dataset lfw --privacy_mechanism metric_privacy --batch_size $BATCH_SIZE --metric_privacy_k $k --dp_epsilon $eps --evaluation_method lfw_validation 
        python evaluate_mechanism.py --dataset CelebA_test --privacy_mechanism metric_privacy --batch_size $BATCH_SIZE --metric_privacy_k $k --dp_epsilon $eps --evaluation_method utility
    done
done