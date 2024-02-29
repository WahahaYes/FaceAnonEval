python process_dataset.py --dataset lfw --privacy_mechanism $1
python evaluate_mechanism.py --dataset lfw --privacy_mechanism $1 --evaluation_method lfw_validation