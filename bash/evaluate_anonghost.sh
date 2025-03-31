BATCH_SIZE=16

for e in -1 0 1 10 100 1000 10000
do

    for t in 45 135  # 0 30 60 90 120 150 180
    do

        clba="anonghost/CelebA_test_eps"$e"_theta"$t
        lfw="anonghost/lfw_eps"$e"_theta"$t


        python evaluate_mechanism.py --dataset CelebA_test --anonymized_dataset $clba --batch_size $BATCH_SIZE --evaluation_method rank_k
        python evaluate_mechanism.py --dataset lfw --anonymized_dataset $lfw --batch_size $BATCH_SIZE --evaluation_method lfw_validation
        python evaluate_mechanism.py --dataset CelebA_test --anonymized_dataset $clba --batch_size $BATCH_SIZE --evaluation_method utility
    done
done