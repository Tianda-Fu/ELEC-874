# Human face generator with GAN/DCGAN
## Introduction
Image acquisition systems have been largely deployed in recent years, due to the reducing cost of manufacturing electronics as well as the increasing demand for collecting customer data. Stronger computation power, machine learning architectures with higher accuracy, and the economic advancement in computer vision systems result in the expansion of the global computer vision market. In 2020, the global computer vision market generated 9.45 billion. Followed by a prediction of $41.1 billion by 2030 which implies a compound annual growth rate of 16% [1]. As the computer vision market expands, a large quantity of images is now available, and the demand for synthesizing and processing images increases. Distinct methods have been investigated to perform video and image editing, such as texture synthesis [2], image inpainting [3], and image stylization [4]. Most of the proposed methods are based on pixels, patches, and low-level features [5] that make high-quality image synthesis, such as photorealistic image stylization, remains a challenging task.

Another popular application for GANs would be applications related to medical imaging. The healthcare industry produces a considerable amount of data every day that can benefit the development of different machine learning algorithms. One of the biggest problems with adapting machine learning in medical imaging is the lack of labelled data [6]. The labelling process can be very time-consuming, and it is heavily dependent on the availability of doctors and other medical professionals. Collecting and constructing a well-labelled dataset for clinical research is a complex, labour-intensive task that might be subject to unexpected issues and biases [7]. 

The training data decides the performance of the supervised machine learning models. These models often require a large amount of labelled data to achieve high accuracy on unseen samples. Data augmentation is commonly used in machine learning and deep learning methods while the number of training samples is limited. Traditional data augmentation methods such as flipping, rotation, and scaling can mislead the generator to learn the distribution of the augmented data [8]. Generative Adversarial Networks (GANs) generate synthetic samples based on a given dataset. In medical imaging, GANs can generate samples that follow the underlying distribution of the original training data [8, 9].

In computer vision, learning and understanding geometric information from source images is a crucial task. The fast advancement of GANs in recent years [10, 11, 12] promotes the development of image synthesis. Through adversarial training,GANs learn the mapping from a latent distribution to the actual training data. Face image synthesis through GANs combines portraits of different people to generate new photorealistic images that exhibit unique identity, expression, and pose [13]. Lately researches on GANs show that in the image synthesizing process, many latent semantics learned by GANs are interpretable [14, 15, 16]. Different units in the generator can represent a specific visual expression or object

## Datasets
- [FFHQ Dataset](https://github.com/NVlabs/ffhq-dataset)

Flickr-Faces-HQ (FFHQ) is a high-quality image dataset of human faces, originally created as a benchmark for generative adversarial networks (GAN). And this dataset was firstly been introduced in the paper of StyleGAN [34]. 

The dataset consists of 70,000 high-quality PNG images at 1024×1024 resolution and contains considerable variation in terms of age, ethnicity and image background. It also has good coverage of accessories such as eyeglasses, sunglasses, hats, etc. The images were crawled from Flickr, thus inheriting all the biases of that website, and automatically aligned and cropped using dlib

|Path|Size|Files|Format|Description|
|:---|:---|:---|:---|:---|
|ffhq-dataset|2.56 TB|210,014||Main folder|
|ffhq-dataset-v2.json |255 MB |1 |JSON |Metadata including copyright info, URLs, etc.|
|images1024x1024 |89.1 GB |70,000 |PNG |Aligned and cropped images at 1024×1024|
|thumbnails128x128 |1.95 GB |70,000 |PNG |Thumbnails at 128×128|
|in-the-wild-images |955 GB |70,000 |PNG |Original images from Flickr|
|tfrecords |273 GB |9 |tfrecords |Multi-resolution data for StyleGAN and StyleGAN2|
|zips |1.28 TB |4 |ZIP |Contentsof each folder as a ZIP archive|

The whole data path is shown as the table above. Considering the computing resources, this project only used a part of all called thumbnails128x128, which is basically resized version of all the images.


Source: https://github.com/NVlabs/ffhq-dataset


- [Anime Face Dataset](https://github.com/bchao1/Anime-Face-Dataset/blob/master/test.jpg)

This dataset contains 63632 high-quality anime faces in the format of JPG images. And the dataset is clean and often used for varying projects with anime faces.

Example images: ![](https://github.com/bchao1/Anime-Face-Dataset/blob/master/test.jpg)

Source: https://github.com/bchao1/Anime-Face-Dataset

## Methods
### DCGAN
GAN mainly consists of two parts of the model - the generator and the discriminator.

The job of the generator is to generate "fake" data that closely resembles the training data, and the job of the discriminator is to determine whether a photo is a real training photo or a fake photo generated by the generator. While training, the generator tries to generate better and better fake photos to fool the discriminator, and the discriminator keeps getting better at spotting and correctly distinguishing between real and fake photos.

The balance of the game is when the generator produces a fake that looks like it came directly from the training data, while the discriminator always guesses with 50% confidence that the generator output is real or a fake.

#### Loss Function
![](https://github.com/marswon0/face_image_generation/blob/main/assets/images/loss_func.JPG)
- x: The data representation of the image.
- D(x): The possibility that the output of the discriminator x comes from the real training data (when x comes from the real training set, the value should be relatively large, when x is generated by the generator, the value should be relatively small) which can be seen as a traditional binary classification.
- z: A latent space vector sampled from the standard normal distribution.
- G(z): Representing the function of the generator mapping from the latent vector z to the data space (the purpose of G is to estimate the distribution from which the training data comes, so that fake samples can be generated from the estimated distribution).
- D(G(z)): The probability that the output of generator G is a real image.

#### Parameters

From the original paper of DCGAN, the authors have recommended some values for the parameters in this network:
- All models are trained with mini-batch stochastic gradient descent (SGD).
- All weights are initialized according to a zero-centerednormal distribution with a standard deviation of 0.02.
- In LeakyReLU, the slopes are all set to 0.2.
- The Adam optimizer was used and hyperparameters were tuned. The suggested learning rate of 0.001 in the base GAN is too high, so use 0.0002 instead.
- Leaving the momentum β1 at the suggested value of 0.9 would causes training oscillations and instability, while lowering it to 0.5 helps stabilize training.

Besides those recommendation parameters, this model also needs some other parameters, and they were set as follows:
|name |value |description|
|:---|:---|:---|
|num_epochs |100 |number of epochs|
|epochs_D |4 |epochs of Discriminator|
|epochs_G |3 |epochs of Generator|
|batch_size |128 |batch size for training|
|image_size |64 |the scale of the input|
|nc |3 |channel of the input|
|nz |100 |the scale of latent vector|



#### Structures
![image](https://user-images.githubusercontent.com/52405215/220776705-ca1545bb-bbe5-4720-b701-25ba73063b17.png)
![image](https://user-images.githubusercontent.com/52405215/220776734-10d22ecd-bee0-4a4b-8c1b-7cbe7db2e822.png)

### StyleGAN2 and StyleGAN3
About the specific function and framework of StyleGAN2 and StyleGAN3, you can see the repository [here](https://github.com/marswon0/face_image_generation). The contributor is also the collaborator of this project.

## Result
### FFHQ
![image](https://user-images.githubusercontent.com/52405215/220784477-e7ed140c-dc7f-4ca2-a2ac-64145e8f7eba.png)

Although StyleGAN2 can generate high-quality images with detailed facial features, the generator suffers from the rotation and translation of the objects. Image below shows some images that did not recognize by the discriminator. We suspect characters in the images suffer from vertical translation, lateral translation, and rotations.

![image](https://user-images.githubusercontent.com/52405215/220784580-216e8e83-f327-4e3e-b43f-5de5a5262ccb.png)

However, the experiment results show that StyleGAN3-r does not have any problem with the texture sticking. Among all the images generated during training, none of the images are distorted by either rotation or translation. 

### Anime Face
To validate the generalization of the proposed models, the Anime Face Dataset is included for testing. The images generated by the proposed GANs are shown below. The architectures and parameters used in the GANs remain the same as previously used in the FFHQ dataset.

![image](https://user-images.githubusercontent.com/52405215/220784943-59f6e35e-3a09-49ff-a9f1-502e545f8f5a.png)

The results obtained from the Anime Face Dataset are similar to the FFHQ Dataset. DCGAN can handle not only human faces, but also anime faces with totally different features in those images. Unlike FFHQ Dataset, anime faces are not based on real scenarios and textures. So, they always have extremely big and colorful eyes as well as colorful hair which are not commonly seen in real life.

Those obvious features make the anime faces less diverse than real human faces, and in some way, they let the GANs much easier to learn. As the result, the anime faces generated from the generator look more “real”.

## Reference
[1] I. Goodfellow et al., "Generative adversarial nets," Neural information processing systems, vol. 27, 2014. 

[2] A. Radford, L. Metz and S. Chintala, "Unsupervised representation learning with deep convolutional generative adversarial networks," arXiv:1511.06434, 2015. 

[3] T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen and T. Aila, "Analyzing and improving the image quality of StyleGAN," arXiv:1912.04958, 2019. 

[4] T. Karras, M. Aittala, S. Laine, E. Härk, J. Hellsten, J. Lehtinen and T. Aila, "Alias-free generative adversarial networks," Advances in Neural Information Processing Systems, vol. 34, 2021. 

[5] T. Karras, S. Laine and T. Aila, "A style-based generator architecture for generative adversarial networks," Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019. 

[6] B. Chao, "Anime-Face-Dataset," 2019. [Online]. Available: https://github.com/bchao1/Anime-Face-Dataset.

[7] A. Nair, "Computer Vision Market by Component, Product, Application, and Vertical (Industrial and Non-Industrial): Global Opportunity Analysis and Industry Forecast, 2020–2030," Allied Market Research, 2022. 

[8] D. Ulyanov, A. Vedaldi, and V. Lempitsky, "Improved Texture Networks: Maximizing Quality and Diversity in Feed-Forward Stylization and Texture Synthesis," In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 6924-6932.

[9] J. Yu, Z. Lin, J. Yang, X. Shen, X. Lu and T. S. Huang, "Generative Image Inpainting With Contextual Attention," Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 5505-5514. 

[10] W. Cho, S. Choi, D. K. Park, I. Shin and J. Choo, "Image-To-Image Translation via Group-Wise Deep Whitening-And-Coloring Transformation," Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 10639-10647. 

[11] X. Wu, K. Xu and P. Hall, "A Survey of Image Synthesis and Editing with Generative Adversarial," in Tsinghua Science and Technology, vol. 22, no. 6, pp. 660-674, 2017, doi: 10.23919/TST.2017.8195348.. 

[12] E. H. Weissler, T. Naumann, T. Andersson, et al., "The role of machine learning in clinical research: transforming the future of evidence generation," Trials 22, p. 537, 2021, https://doi.org/10.1186/s13063-021-05489-x. 

[13] P. Shah, F. Kendall, S. Khozin, et al., "Artificial intelligence and machine learning in clinical development: a translational perspective," npj Digit. Med., vol. 2, p. 69, 2019, https://doi.org/10.1038/s41746-019-0148-3. 

[14] N. Tran, V. Tran, N. Nguyen, T. Nguyen and N. Cheung, "On Data Augmentation for GAN Training," arXiv, 2020, arXiv:2006.05338. 

[15] C. Bowles et al., "GAN Augmentation: Augmenting Training Data using Generative Adversarial Networks," arXiv, 2018, arXiv:1810.10863. 

[16] S. Tripathy, J. Kannala and E. Rahtu, "ICface: Interpretable and Controllable Face Reenactment Using GANs," Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2020, pp. 3385-3394. 
