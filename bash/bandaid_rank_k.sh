BATCH_SIZE=32

for eps in -1 0 1 10 50 100 200 400 600 800 1000 # -1 0 1 10 100 1000 10000
do
    python evaluate_mechanism.py --dataset CelebA_test --privacy_mechanism dtheta_privacy --batch_size $BATCH_SIZE --theta 0 --dp_epsilon $eps --evaluation_method rank_k
    python evaluate_mechanism.py --dataset CelebA_test --privacy_mechanism identity_dp --batch_size $BATCH_SIZE --dp_epsilon $eps --evaluation_method rank_k
done