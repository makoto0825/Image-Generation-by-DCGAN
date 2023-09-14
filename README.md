
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

## 1.3 DCGAN(Deep Convolutional GAN)
DCGAN is improved the version of GAN. The proposed model is based on the DCGAN model. This model uses  CNN techniques into the GAN framework. In the generator, it uses transpose convolutional layers to perform upsampling. 

<p align="center">
  <img src="https://github.com/makoto0825/Image-Generation-by-DCGAN/assets/120376737/8c0dde35-37b1-4efd-911f-01eecaf3e883" />
</p>
The upsampling using the Transpose Convolutional layer is a key feature of DCGAN. There is a 2x2 input data. In upsampling, we increase the size of the matrix by doubling the rows and columns. Next, for each pixel in the kernel, we output the result by performing multiplication while sliding it. This process is repeated until the desired size is achieved, and then any unnecessary parts are removed. By performing these operations, the size gradually expands.

In this project, The DCGAN model presented by Keras is used as the baseline model. (https://keras.io/examples/generative/dcgan_overriding_train_step/) and I improved the baseline model. The difference between baseline model and proposal model(improved model) is following.

**Discriminator**
<p align="center">
  <img src="https://github.com/makoto0825/Image-Generation-by-DCGAN/assets/120376737/5a36d7f7-9106-4295-9913-a9b61232ca4c" />
</p>
I tried several improvements to prevent overfitting. I added multiple dropout layers to the model. Additionally, a middle dense layer also was put. Moreover, Instead of using Flatten, I used GlobalAveragePooling2D. Flatten simply flattens the feature maps into a 1-dimensional vector, on the other hand, GlobalAveragePooling2D computes the average value for each channel of the feature maps and extracts them as feature vectors. By using GlobalAveragePooling2D, I can reduce the number of parameters compared to Flatten, it  results in a simpler model and expected prevention of overfitting.

**Generator**
<p align="center">
  <img src="https://github.com/makoto0825/Image-Generation-by-DCGAN/assets/120376737/69ef821d-1f09-4c44-b15e-e80fdac6c5bd" />
</p>
I have added BatchNormalization layers after the convolutional layers. This performs normalization based on the mean and variance of each batch. Additionally, I have changed the activation function from LeakyReLU to ReLU. While the baseline model's discriminator used LeakyReLU for all layers except the final one, the original DCGAN paper used ReLU, so I adopted it in my model. Furthermore, I set the learning rate for the generator to be twice that of the discriminator. This is because, in general, the discriminator tends to have better accuracy than the generator, so to strike a balance, I set the learning rate to be twice as high for the generator.
