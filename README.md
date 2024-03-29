# Dynamic Attention GAN for Image Inpainting
A PyTorch reimplementation for the paper [DAM-GAN : Image Inpainting using Dynamic Attention Map based on Fake Texture Detection](https://arxiv.org/abs/2204.09442). (ICASSP 2022)

## Abstract 
Deep neural advancements have recently brought remarkable image synthesis performance to the field of image inpainting. The adaptation of generative adversarial networks(GAN) in particular has accelerated significant progress in high-quality image reconstruction. However, although many notable GAN-based networks have been proposed for image inpainting, still pixel artifacts or color inconsistency occur in synthesized images during the generation process, which are usually called fake textures. To reduce pixel inconsistency disorder resulted from fake textures, we introduce a GAN-based model using dynamic attention map (DAM-GAN). Our proposed DAM-GAN concentrates on detecting fake texture and products dynamic attention maps to diminish pixel inconsistency from the feature maps in the generator. Evaluation results on CelebA-HQ and Places2 datasets with other image inpainting approaches show the superiority of our network.

## Instruction

<p align="center"><img src="samples/sample_img.png" width="720"\></p>

We have two types of masks (fixed/random). We have 4,000 samples of random masks from [Here](https://drive.google.com/file/d/120eCnAaK-BHdkLlw_LZ89rvTCgkU0kep/view?usp=sharing). Default random mask directory's name is 'rand_masks'.

```
$ unzip dam_rand_masks.zip
```

In **'configs/config.yaml**, you can change the random mask directory from **'rand_mask_path'**.


## Acknowledgement
 + Most functions are brought from Contextual Attention(https://github.com/daa233/generative-inpainting-pytorch). 
 + This work was supported by Institute of Information & communications Technology Planning Evaluation(IITP) grant funded by the Korea government(MSIT) (No.B0101-15-0266, Development of High Performance Visual BigData Discovery Platform for Large-Scale Realtime Data Analysis), (No.2017-0-00897, Development of Object Detection and Recognition for Intelligent Vehicles) and (No.2018-0-01290, Development of an Open Dataset and Cognitive Processing Technology for the Recognition of Features Derived From Unstructured Human Motions Used in Self-driving Cars)