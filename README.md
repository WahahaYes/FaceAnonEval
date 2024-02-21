# FaceAnonEval
A codebase to evaluate SOTA in face anonymization.

## Apptainer instructions
Run one of the following commands to initialize an apptainer image:
```
apptainer build --sandbox faceanoneval docker://azimibragimov/faceanoneval:gpu # installs gpu version of codebase. 
apptainer build --sandbox faceanoneval docker://azimibragimov/faceanoneval:cpu # installs cpu version of codebase. 

```

Now, ensure that the folder directory looks like this: 
```
- root
	- AnnonymzedDataset	# Folder where results of process_dataset.py are stored. Initially it is an empty folder.
	- Datasets  		# Contains datasets
		- lfw
		- CelebA
	- Results		# Folder where results of evaluate_mechanisms.py are stored. Initially it is an empty folder.
	- Models		# Folder containing "buffalo_l" weights from insightface: https://github.com/deepinsight/insightface/releases/tag/v0.7
	- faceannoneval		# Apptainer sandbox instance

```



Then, run the following to initalize a shell within the image
```
apptainer shell --nv --contain \
	--bind /path/tp/AnnonymizedDataset:/workspace/FaceAnonEval/Anonymized\ Datasets \
	--bind /path/to/Datasets:/workspace/FaceAnonEval/Datasets \
	--bind /path/to/Results:/workspace/FaceAnonEval/Results 
	--bind /path/to/models:/home/<USER>/.insightface/models/buffalo_l \
	--pwd /workspace/FaceAnonEval/ faceanoneval/

```

Now you can run commands within the shell. For example: 
```
python process_dataset.py --dataset lfw
python evaluate_mechanism.py --dataset lfw
```


