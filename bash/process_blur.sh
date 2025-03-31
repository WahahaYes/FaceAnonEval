BATCH_SIZE=16

for b in gaussian_blur
do

    for k in 5 11 15 21 25
    do
        python process_dataset.py --dataset CelebA_test --privacy_mechanism $b --batch_size $BATCH_SIZE --blur_kernel $k
        python process_dataset.py --dataset lfw --privacy_mechanism $b --batch_size $BATCH_SIZE --blur_kernel $k

        python evaluate_mechanism.py --dataset CelebA_test --privacy_mechanism $b --batch_size $BATCH_SIZE --blur_kernel $k --evaluation_method rank_k
        python evaluate_mechanism.py --dataset lfw --privacy_mechanism $b --batch_size $BATCH_SIZE --blur_kernel $k --evaluation_method lfw_validation
        python evaluate_mechanism.py --dataset CelebA_test --privacy_mechanism $b --batch_size $BATCH_SIZE --blur_kernel $k --evaluation_method utility
    done
done