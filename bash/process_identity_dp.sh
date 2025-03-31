BATCH_SIZE=32

for i in 1 10 50 100 200 400 600 800 1000 1200 1400
do
    python process_dataset.py --dataset CelebA_test --privacy_mechanism identity_dp --batch_size $BATCH_SIZE --dp_epsilon $i
    python process_dataset.py --dataset lfw --privacy_mechanism identity_dp --batch_size $BATCH_SIZE --dp_epsilon $i

    python evaluate_mechanism.py --dataset CelebA_test --privacy_mechanism identity_dp --batch_size $BATCH_SIZE --dp_epsilon $i --evaluation_method rank_k
    python evaluate_mechanism.py --dataset lfw --privacy_mechanism identity_dp --batch_size $BATCH_SIZE --dp_epsilon $i --evaluation_method lfw_validation
    python evaluate_mechanism.py --dataset CelebA_test --privacy_mechanism identity_dp --batch_size $BATCH_SIZE --dp_epsilon $i --evaluation_method utility
done