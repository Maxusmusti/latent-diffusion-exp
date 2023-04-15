# Latent Diffusion Models for Conditional Image Generation

## Authors

Rami Matar (rhm2142), Mustafa Eyceoz (me2680), Justin Lee (jjl2245)


## Project Summary

Diffusion models are very effective learners of data distributions that learn to denoise Gaussian-blurred images and reverse the steps of the diffusion process. The diffusion process is essentially a noising process in which Gaussian noise is applied to an input image repeatedly for T steps. Initially [2], diffusion models trained a sequence of T autoencoders to learn the denoising process in the pixel space. A much more computationally efficient variant of the diffusion model, known as the Latent Diffusion Model (LDM) works by performing the diffusion process within the latent space of a pre-trained autoencoder [1]. These models have shown very competitive results on many image synthesis tasks, and can further allow for learning many conditional distributions simultaneously without much change to training.

The goal of our project is to build a latent diffusion model for conditional image synthesis. Naturally, the learned distribution of a diffusion model is effective at allowing the modification of images in more semantically meaningful ways. We plan to train the model to generate images with a conditioned style or text-description. Our first goal will be to learn to generate images based on a style; to do that, we inject the embedding generated from a (potentially separate) encoder into different parts of the denoising networks through cross-attention. Our second goal will be to generate images from text input using a similar process with a pretrained language encoder in order to obtain an embedding which can be treated similarly. If time allows, we may explore the relationship between LDMs and a model like CLIP.

## Approach

- **Architecture:** We will utilize a pre-trained image autoencoder and learn the reverse diffusion process inside of its latent space. Our general architecture will involve an encoder to process the input into the latent space, followed by a sequence of scheduled Gaussian noise to be added to the image until it effectively looks like Gaussian noise. To learn the denoising, we define a U-Net network to learn to reconstruct the image from its noisy representation. Conditional inputs will be processed through a pretrained domain-specific encoder and injected into the U-Net layers before and during denoising through cross attention layers. After the T denoising U-Nets, the reconstructed latent image representation is reconstructed into image space through the pre-trained decoder.  
- **Training objective:** We will train the model in two steps. In the first step, we train with supervised data and a reconstruction loss. In the second, we will expand the work of the original paper [1] by a self-supervised training approach with a KL divergence regularization term in the loss.
- **Dataset:** We will use a subset of LAION-400M dataset, a massive dataset popular for training large models like diffusion models. We will also use a large image-caption dataset.


## Tools

- Python3 environment
- Python libraries: 
    - ```TK```
    - ```TK```
## Usage
**Run demo**

```
$ python3 TK.py
```

**Description**

- The demo scripts do the following...
    - TK
    - TK
    
## References

1. Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
2. Dhariwal, Prafulla, and Alexander Nichol. "Diffusion models beat gans on image synthesis." Advances in Neural Information Processing Systems 34 (2021): 8780-8794.