BATCH_SIZE=16

# for k in 1 2 3 4 5 6
# do

#     clba="codec_k"$k


#     python evaluate_mechanism.py --dataset codec --anonymized_dataset $clba --batch_size $BATCH_SIZE --evaluation_method rank_k --compare_exact_query True --overwrite_embeddings True
#     python evaluate_mechanism.py --dataset codec --anonymized_dataset $clba --batch_size $BATCH_SIZE --evaluation_method utility
# done

for eps in -1 0 1 10 50 100 500 1000
do

    clba="codecpca_eps"$eps


    python evaluate_mechanism.py --dataset codecpca --anonymized_dataset $clba --batch_size $BATCH_SIZE --evaluation_method rank_k --compare_exact_query True --overwrite_embeddings True
    python evaluate_mechanism.py --dataset codecpca --anonymized_dataset $clba --batch_size $BATCH_SIZE --evaluation_method utility --overwrite_embeddings True
done

for theta in 30 60 90 120 150 180
do

    clba="codecpca_theta"$theta


    python evaluate_mechanism.py --dataset codecpca --anonymized_dataset $clba --batch_size $BATCH_SIZE --evaluation_method rank_k --compare_exact_query True --overwrite_embeddings True
    python evaluate_mechanism.py --dataset codecpca --anonymized_dataset $clba --batch_size $BATCH_SIZE --evaluation_method utility --overwrite_embeddings True
done