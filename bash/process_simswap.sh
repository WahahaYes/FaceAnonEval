BATCH_SIZE=16

for i in ssim_similarity ssim_dissimilarity all_to_one random
do
    python process_dataset.py --dataset CelebA_test --privacy_mechanism simswap --batch_size $BATCH_SIZE --faceswap_strategy $i
    python process_dataset.py --dataset lfw --privacy_mechanism simswap --batch_size $BATCH_SIZE --faceswap_strategy $i

    python evaluate_mechanism.py --dataset CelebA_test --privacy_mechanism simswap --batch_size $BATCH_SIZE --faceswap_strategy $i --evaluation_method rank_k 
    python evaluate_mechanism.py --dataset lfw --privacy_mechanism simswap --batch_size $BATCH_SIZE --faceswap_strategy $i --evaluation_method lfw_validation 
    python evaluate_mechanism.py --dataset CelebA_test --privacy_mechanism simswap --batch_size $BATCH_SIZE --faceswap_strategy $i --evaluation_method utility
done
