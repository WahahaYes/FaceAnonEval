# FaceAnonEval
A codebase to evaluate SOTA in face anonymization.

## Apptainer instructions
Run one of the following commands to docker an docker image:
```
docker build -t faceanoneval:gpu .
```

Then, run the following to initalize a shell within the image
```
docker run --gpus all -it faceanoneval:gpu
```

Now you can run commands within the shell. For example: 
```
python process_dataset.py --dataset lfw
python evaluate_mechanism.py --dataset lfw
```


