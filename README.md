
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

## 1.4 Dataset
<p align="center">
  <img src="https://github.com/makoto0825/Image-Generation-by-DCGAN/assets/120376737/e8b99ac7-8424-4602-b87c-d371969bf301" />
</p>
I used dataset of anime face images which comes from Kaggle(https://www.kaggle.com/datasets/splcher/animefacedataset).

## 1.5 Result
<p align="center">
  <img src="https://github.com/makoto0825/Image-Generation-by-DCGAN/assets/120376737/e6dafe01-d953-4287-8be8-3e40a29037be" />
</p>
This is the output of the baseline model. The training was executed for 200 epochs. In epoch 11, I can see faces but the model collapse was happended. Furthermore, in epoch 200, there is no output.

<p align="center">
  <img src="https://github.com/makoto0825/Image-Generation-by-DCGAN/assets/120376737/faed12e6-0ab7-4320-bcde-91ef5bf5b3f5" />
</p>
On the other hand, the proposed model was able to generate various types of faces. Even in the final epoch.

# 2.Code part. 
In this section, I will show the code and explain it. My architecture is refered to Keras and improved it. 
Keras: (https://keras.io/examples/generative/dcgan_overriding_train_step/)

# 2.1 Development Environment
I used Google Colaboratory as our Integrated Development Environment (IDE). The reason for this choice is that Colaboratory provides high-performance GPUs, which are beneficial for training deep learning models. Specifically, I utilized the NVIDIA A100 GPU for our tasks. For programming, I opted to use Python and selected Keras and Tensorflow as our deep learning libraries for training.

# 2.2 Image Preprocessing
In conducting the training, I performed several preprocessing steps on the image data. First, I applied a process to unify all images to 64x64 resolution because GAN cannot properly learn if the input image resolution is not consistent. Second, I normalized each pixel value to be in the range of -1 to 1. In DCGAN, the last activation function of the Generator network is tanh, which results in generated images having a scale of [-1, 1]. Therefore, I subtracted the mean pixel value of 127.5 and divided by the same value to convert the value range to [-1, 1]. This scaling is expected to normalize the brightness values of the images and stabilize the model's learning. Fllowing shows the program code for these preprocessing steps.
<p align="center">
  <img src="https://github.com/makoto0825/Image-Generation-by-DCGAN/assets/120376737/a1b7fe45-f437-436e-8577-0f45a804f4f0" />
</p>

# 2.2 Generator
The generator of proposal model is composed of a combination of fully connected layers, transpose convolution layers, batch normalization layers, and activation functions, using the Sequential function. The Fllowing shows Generator.
<p align="center">
  <img src="https://github.com/makoto0825/Image-Generation-by-DCGAN/assets/120376737/7f794e88-1aa3-4fd6-8dfd-792487eff639" />
</p>

Firstly, a fully connected layer is added to the model, which takes random noise as input and is used to convert it into a matrix of size 16x16x128. Then, transpose convolutions are repeated, gradually increasing the size of the features (upsampling), and generating images. Finally, this process is repeated until the image reaches a resolution of 64x64. Batch normalization is utilized in the transpose convolution layers. Batch normalization is a method that normalizes the data distribution in each layer (to have a mean of 0 and a variance of 1), which enhances the convergence of learning and reduces the dependence on initial parameter values. After normalization, the ReLU (Rectified Linear Unit) activation function is used. ReLU outputs 0 for inputs less than or equal to 0 and the input value for inputs greater than 0. It has a very simple structure, low computational cost, and can handle the problem of gradient disappearance. The gradient disappearance problem refers to the phenomenon in which the gradient gradually decreases as the number of layers in deep learning increases, and hardly updates when it reaches the first layer during backpropagation. The Fllowing shows the ReLU.
<p align="center">
  <img src="https://github.com/makoto0825/Image-Generation-by-DCGAN/assets/120376737/499fef92-ff9e-4b66-a704-336ae9bf6c6c" />
</p>

After repeating these transpose convolution processes, only the final output layer applies the activation function Tanh to generate learning. The Tanh function outputs in the range of -1 to 1. The reason for using the Tanh function is to limit the generated image's pixel values to the range of -1 to 1, the same as the dataset's image.The Fllowing shows the Tanh function.
<p align="center">
  <img src="https://github.com/makoto0825/Image-Generation-by-DCGAN/assets/120376737/9555d2cb-936d-4dab-8de4-1f032d77e383" />
</p>

# 2.3 Discriminator
The discriminator in proposal model is composed of convolutional layers, GAP layer, Dropout layer, activation functions, and fully-connected layers, combined using Keras' Sequential function. The convolution is repeated several times to gradually reduce the feature size and ultimately determine whether the image data is real or fake (downsampling).The Fllowing shows the Discriminator.
<p align="center">
  <img src="https://github.com/makoto0825/Image-Generation-by-DCGAN/assets/120376737/9d6edd83-0d33-4fa5-a8a3-f681b674f59e" />
</p>

In #Convolution 1 of the picture, convolution is performed while sliding by 2. Then, the LeakyReLU activation function is applied. LeakyReLU defines the negative slope with the argument "a," which allows the output to differ even when the input is less than 0.The Fllowing shows the LeakyReLU function.
<p align="center">
  <img src="https://github.com/makoto0825/Image-Generation-by-DCGAN/assets/120376737/53f1e68c-1238-443d-9392-3c70394433fa" />
</p>

Similarly, convolution is performed with the same process in #Convolution 2, and then Dropout processing is applied with model.add(Dropout(0.25)). This is the original idea of the proposed model.In #Convolution 3, convolution and LeakyReLU function are applied similarly. Then, the GAP layer is added with model.add(GlobalAveragePooling2D()). This is also the original idea, which compresses the output of the previous convolutional layer into one dimension by taking the average.This can also help suppress learning. Then, a fully-connected layer with 256 units is added in #fully-connected layer1, and LeakyReLU function and Dropout processing are applied. Finally, an output layer with one unit is added in #fully-connected layer2, and the sigmoid function is applied to output the value. The sigmoid activation function is a function that scales input values between 0 and 1, and is commonly used in the output layer of neural networks. The Fllowing shows the sigmoid function.
<p align="center">
  <img src="https://github.com/makoto0825/Image-Generation-by-DCGAN/assets/120376737/798aaa61-d33a-483d-94c5-194fa0e54c13" />
</p>

# 2.3 Training 
The fllowing picture is parameters. The model utilized an optimizer called Adam. Additionally, the proposed model employed a training ratio of 2:1 between the generator and the discriminator. This is becuse the discriminator tends to learn at a faster pace compared to the generator. In such situations, the discriminator can easily detect fake samples at a high rate, hindering the effective learning of the generator. To avoid this scenario, the learning rate for the generator was increased. Furthermore, the training was conducted for 200 epochs with a total of 30,000 iterations under these specified conditions.
<p align="center">
  <img src="https://github.com/makoto0825/Image-Generation-by-DCGAN/assets/120376737/e8a1628b-7280-40ab-ae64-9d5337fe483f" />
</p>

The model uses binary cross-entropy as loss function. Binary cross-entropy is a measure of how "different" two probability distributions are. The fllowing images represents the formula for binary cross-entropy. Let P be the true probability distribution and Q be the estimated probability distribution. The reason why the binary cross-entropy loss function is used is to evaluate the difference between the predicted distribution of the discriminator and the true distribution of the actual data. If we can train the generator to approach the true data distribution (by minimizing the cross-entropy), it means that the generated images approximate the real images. 
<p align="center">
  <img src="https://github.com/makoto0825/Image-Generation-by-DCGAN/assets/120376737/11598330-aaaf-45ea-8841-e685c8c248ed" />
</p>

Training is conducted in df train function. The fllowing picture is df train function. Using the provided arguments (batch size, number of epochs, number of image saves), the function performs the training of the GAN model. The generator generates images using generator.predict(noise). The discriminator is trained on actual image data with corresponding labels set as 1 using discriminator.train_on_batch(imgs, np.ones((half_batch, 1))), and the loss function is obtained. The discriminator is also trained on fake images generated by the generator, with corresponding labels set as 0 using discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1))), and the loss function is obtained. The two loss functions are summed and divided by 2 to obtain the final discriminator loss function. The generator's loss function is trained and obtained using combined_model.train_on_batch(noise, valid_y). These pieces of information are logged and outputted, and the def_image function is called to save the images.
<p align="center">
  <img src="https://github.com/makoto0825/Image-Generation-by-DCGAN/assets/120376737/b5954439-ef45-4414-b77e-89072b579799" />
</p>

# 2.4	Save images
The images generated by the generator are saved one by one in the def save_imgs function. First, the noise is defined, which specifies the number of images to generate per epoch. In this case, I generate 25 images per epoch, arranged in a 5x5 matrix (r rows, c columns). Once the 25 images are generated by the generator, their pixel values are converted to the range of [0, 1] (scaling). Then, using a loop, each generated image is saved to the specified location.
<p align="center">
  <img src="https://github.com/makoto0825/Image-Generation-by-DCGAN/assets/120376737/6d4db597-137b-4d3f-89c7-32229b8b4d54" />
</p>

