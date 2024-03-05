### SimSwap Integration
We use SimSwap as a face swapping back-end for many of our implemented privacy mechanisms.  This includes face swapping methods and our implementation of IdentityDP.  

We provide a subset of the SimSwap codebase needed to run inference in our codebase.  Please refer to their project page and acknowledge their licensing agreements:
- SimSwap's home page: https://github.com/neuralchen/SimSwap
- SimSwap's License: CC BY-NC 4.0 DEED Attribution-NonCommercial 4.0 International
---
To run our methods based on SimSwap, please download the pretrained model files available [here](https://github.com/ethanrwilson1998/FaceAnonEval/releases/tag/models).

Unzip `people.zip` into `src/privacy_mechanisms/simswap/checkpoints`.  The resulting file structure should look like this:
```
+---checkpoints
    +---people
            519_net_D1.pth
            519_net_D2.pth
            519_net_G.pth
            iter.txt
            latest_net_D1.pth
            latest_net_D2.pth
            latest_net_G.pth
            loss_log.txt
            opt.txt
        +---samples
        +---summary
        +---web
            +---images
```

Place `netArc.pt` into `src/privacy_mechanisms/simswap/arcface_model`.  The resulting file structure should be:
```
+---arcface_model
        netArc.pt
```
---
If you use any method based on SimSwap in an academic publication, please cite their work as well:
```
@inproceedings{chen_simswap_2020,
    author = {Renwang Chen and Xuanhong Chen and Bingbing Ni and Yanhao Ge},
    title = {SimSwap: An Efficient Framework For High Fidelity Face Swapping},
    booktitle = {{MM} '20: The 28th {ACM} International Conference on Multimedia},
    year = {2020}
}
```
