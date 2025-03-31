BATCH_SIZE=16

for b in 1 2 4 8 16
do

    for eps in 1.0 2.0 3.0 5.0 7.0 10.0 15.0
    do
        python process_dataset.py --dataset CelebA_test --privacy_mechanism pixel_dp --batch_size $BATCH_SIZE --pixel_dp_b $b --dp_epsilon $eps
        python process_dataset.py --dataset lfw --privacy_mechanism pixel_dp --batch_size $BATCH_SIZE --pixel_dp_b $b --dp_epsilon $eps

        python evaluate_mechanism.py --dataset CelebA_test --privacy_mechanism pixel_dp --batch_size $BATCH_SIZE --pixel_dp_b $b --dp_epsilon $eps --evaluation_method rank_k 
        python evaluate_mechanism.py --dataset lfw --privacy_mechanism pixel_dp --batch_size $BATCH_SIZE --pixel_dp_b $b --dp_epsilon $eps --evaluation_method lfw_validation
        python evaluate_mechanism.py --dataset CelebA_test --privacy_mechanism pixel_dp --batch_size $BATCH_SIZE --pixel_dp_b $b --dp_epsilon $eps --evaluation_method utility
    done
done