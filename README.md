
<p align= center">
  <img src="https://github.com/makoto0825/Image-Generation-by-DCGAN/assets/120376737/43472a41-c4c7-4a35-b60c-10b832966228" />
</p>

# 1.methodology part. 
In this part, I will explain about some methodology or notions in my proposal system.
## 1.1 What is GAN(Generative Adversarial Network)?
<p align="center">
  <img src="https://github.com/makoto0825/Image-Generation-by-DCGAN/assets/120376737/26b0a439-68e8-498d-949d-2c5f8cc5a21a" />
</p>
<p>
Gan is one of the techniques commonly used in the field of image processing. It is used in various domains such as image generation, image transformation, video generation, and more. In this project, I tried to generate anime face images by using this method.

## 1.2 GAN’s mechanism
<p align="center">
  <img src="https://github.com/makoto0825/Image-Generation-by-DCGAN/assets/120376737/872fc9ac-0ca7-4d56-bd2e-226e25af58db" />
</p>
GAN consists of two architectures: the generator and the discriminator. The generator aims to generate image that resembles the training data. On the other hand the discriminator aims to distinguish between real and generated image. Through iterative tasks performed by these two architectures, the generator ultimately can generate high-quality anime images

<p align="center">
  <img src="https://github.com/makoto0825/Image-Generation-by-DCGAN/assets/120376737/db9e9652-060c-4678-b4d1-122b6c917664" />
</p>
This image shows sample of two architectures. The generator gradually expands features through multiple dense layers and eventually performs a reshape operation to output an image. On the other hand, the discriminator takes an input image through multiple dense layers and outputs a single value, which determining whether the input image is real or fake. The activation function used in both architectures is ReLU.
