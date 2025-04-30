
cmd1="python process_dataset.py --dataset lfw --batch_size 16"
cmd2="python evaluate_mechanism.py --dataset lfw --evaluation_method lfw_validation --batch_size 16"

# no mechanism
$cmd2 --anonymized_dataset ../Datasets/lfw 

# pixeldp strong
$cmd1 --privacy_mechanism pixel_dp --pixel_dp_b 16 --dp_epsilon 1
$cmd2 --privacy_mechanism pixel_dp --pixel_dp_b 16 --dp_epsilon 1
# pixeldp weak
$cmd1 --privacy_mechanism pixel_dp --pixel_dp_b 4 --dp_epsilon 15
$cmd2 --privacy_mechanism pixel_dp --pixel_dp_b 4 --dp_epsilon 15
# metricsvd strong
$cmd1 --privacy_mechanism metric_privacy --metric_privacy_k 4 --dp_epsilon 1
$cmd2 --privacy_mechanism metric_privacy --metric_privacy_k 4 --dp_epsilon 1
# metricsvd weak
$cmd1 --privacy_mechanism metric_privacy --metric_privacy_k 8 --dp_epsilon 10
$cmd2 --privacy_mechanism metric_privacy --metric_privacy_k 8 --dp_epsilon 10

for THETA in 0 45 60 90 135 150
do
    # our method varying theta
    $cmd1 --privacy_mechanism dtheta_privacy --theta $THETA --dp_epsilon -1
    $cmd2 --privacy_mechanism dtheta_privacy --theta $THETA --dp_epsilon -1
done

for EPS in 1 10 50 100 200
do
    # our method varying theta
    $cmd1 --privacy_mechanism dtheta_privacy --theta 0 --dp_epsilon $EPS
    $cmd2 --privacy_mechanism dtheta_privacy --theta 0 --dp_epsilon $EPS
    # identitydp
    $cmd1 --privacy_mechanism identity_dp --dp_epsilon $EPS
    $cmd2 --privacy_mechanism identity_dp --dp_epsilon $EPS
done

# random face swapping
$cmd1 --privacy_mechanism simswap --faceswap_strategy random
$cmd2 --privacy_mechanism simswap --faceswap_strategy random
