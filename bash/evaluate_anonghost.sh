BATCH_SIZE=16

for e in -1 0 1 10 50 100 200 1000
do
    clba="anonghost/CelebA_test_eps"$e"_theta0"
    lfw="anonghost/lfw_eps"$e"_theta0"    


    python evaluate_mechanism.py --dataset CelebA_test --anonymized_dataset $clba --batch_size $BATCH_SIZE --evaluation_method rank_k
    python evaluate_mechanism.py --dataset lfw --anonymized_dataset $lfw --batch_size $BATCH_SIZE --evaluation_method lfw_validation
    python evaluate_mechanism.py --dataset CelebA_test --anonymized_dataset $clba --batch_size $BATCH_SIZE --evaluation_method utility
done

for t in 0 30 60 90 135 150
do
    clba="anonghost/CelebA_test_eps-1_theta"$t
    lfw="anonghost/lfw_eps-1_theta"$t  

    python evaluate_mechanism.py --dataset CelebA_test --anonymized_dataset $clba --batch_size $BATCH_SIZE --evaluation_method rank_k
    python evaluate_mechanism.py --dataset lfw --anonymized_dataset $lfw --batch_size $BATCH_SIZE --evaluation_method lfw_validation
    python evaluate_mechanism.py --dataset CelebA_test --anonymized_dataset $clba --batch_size $BATCH_SIZE --evaluation_method utility
done