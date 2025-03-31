BATCH_SIZE=16

for THETA in 0 # 0 30 60 90 120 150 180
do

    for i in 50 200 400 600 800 # -1 0 1 10 100 1000 10000
    do
        python process_dataset.py --dataset CelebA_test --privacy_mechanism dtheta_privacy --batch_size $BATCH_SIZE --theta $THETA --dp_epsilon $i
        python process_dataset.py --dataset lfw --privacy_mechanism dtheta_privacy --batch_size $BATCH_SIZE --theta $THETA --dp_epsilon $i

        python evaluate_mechanism.py --dataset CelebA_test --privacy_mechanism dtheta_privacy --batch_size $BATCH_SIZE --theta $THETA --dp_epsilon $i --evaluation_method rank_k
        python evaluate_mechanism.py --dataset lfw --privacy_mechanism dtheta_privacy --batch_size $BATCH_SIZE --theta $THETA --dp_epsilon $i --evaluation_method lfw_validation
        python evaluate_mechanism.py --dataset CelebA_test --privacy_mechanism dtheta_privacy --batch_size $BATCH_SIZE --theta $THETA --dp_epsilon $i --evaluation_method utility
    done
done