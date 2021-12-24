# UOB_NC_CourseWork2021

## Introduction

​	The task we offered this semester is using deep learning to do the semantic segmentation of magnetic resonance (MR) images.

### Dataset

​	Basically, the offered dataset contains 200 cardiovascular magnetic resonance (CMR) images, which were split into three parts: 50% for training set, including 100 CMR images and 100 ground truth mask images, 10% for validation set, with 20 CMR images and 20 ground truth mask images, and there are 40% for test set, which contains 40 CMR images.

### Machine learning task

​	Unlike classic semantic segmentation, medical image segmentation is much more challenging [4]. Due to the environment changing or cell movement, cell deformation is a common situation. What’s more, the defects of medical imaging equipment and the structure of lesion may cause some discontinuity issue of segmentation boundary. 

​	Therefore, comparing to classic semantic segmentation task, medical image segmentation task expects our model is highly generalized and performs higher accuracy on segmentation. Our main task is trying to get a well-generalized model which can segment each CMR image into segmented mask accurately including 4 regions: the background region (black), the left ventricle (LV) region (white), the myocardium (Myo) region (White gray) and the right ventricle (RV) region (dark gray).

### Relevant work

​	Normally, the main approach of semantic segmentation is using convolution neural network to classify each pixel by allocate different labels. While some popular models were given by professor like FCN, DeepLab and SegNet, we found that Unet is one of the best performance model on medical image segmentation which recognized by mainstream of academia [6]. 

​	One of the most innovation design in the Unet is called skip connection which inspired by a skip architecture mentioned in Full Convolution Network that combines different prediction information from various layers. In this coursework, we built the Unet and Unet 3+, trained the neural network based on the given data and gained the best performance model by data augmentation and fine-tuning the hyperparameters at the end. Considering the cost of calculation, the difficulty of implementation and the timeliness, we decided to build SegNet to be the control group of the Unet.

### Aim to achieve

​	Hence, our trained model is ideally to be efficient both in terms of memory and computational time during inference [1]. Moreover, we expects our model have great robustness that the generated mask images should be segmented accurately and smoothly even though sometimes the original cell image got some extreme deformations.

## Implementation

​	Totally, there are three models implemented by our group.

### SegNet

​	At first, we implemented the SegNet model consists of two symmetrical part which contains one encoder and one decoder. The encoder network is topologically identical to the convolution layers in VGG16[2] without fully connected layers which can reduce the cost of calculation effectively. Based on that, each layer in encoder adds batch normalized, using ReLU to be the activation function and Max-pooling layer follows. 

​	Although the architecture of decoder is correspond to the encoder, the key operation for the decoder is up-sampling by using Max-pooling indices to restore the original resolution of input and a softmax layer will be added to implement pixel-wise classification.

### Unet

​	Let us turn our attention to Unet. Figure 2.1 illustrates the architecture of Unet which is a kind of auto-encoder[5], consists of a contracting path and a expansive path. At the left side is a contracting path, each contracting block consists of two repeated application of convolutions follow with one Max-pooling for downsampling which doubles the number of feature channels, each convolution layer follows by a activation function (ReLU). 

​	Every step in expansive path which at the right side using the inverse convolution to upsample the feature map that halves the numbers of feature channels, each follows by a ReLU. And the feature information in contracting path can be transmitted by skip connection from the same-scale.

​	In our Unet, a 3*3 convolution kernel and a 2*2 max pooling operation with stride 1 and stride 2 respectively we utilized for downsampling in contracting path. For expansive path, same convolution operation for feature extraction with the same stride while a 2*2 inverse convolution operation with stride 2 for each unsampling. Four layers (96*96) would be classified by 1*1 convolution repeated four times in the last layer.

<img src=".\res_pic\UnetStructure.png" style="zoom:50%;" />

<table>
    <center>
    Figure 2.1 Unet Structure
    </center>
</table>

​	We tried our best to keep enough information for feature maps. For instance, we reduced the depth of the network to four unlike [5] which got five steps since the information of the feature map (6*6) would be quite limited if we applied the offered images (96*96) as the input of the Unet with depth five. Similarly, the padding included in each convolution avoided size reduction for the same reason, which means that only max pooling operation would change the size of images.

### Unet 3+

​	While Unet 2+ gets a progress on medical image segmentation accuracy, the nested and dense skip connection architecture rises the quantity of parameters. However, the idea of combining different features of each layer inspires the Unet 3+.

​	Comparing to Unet, Unet 3+ utilizing bilinear interpolation for upsampling instead of inverse convolution in Unet, which is a data augmentation approach improves boundary smoothness of image .

​	What’s more, the biggest difference between Unet and Unet 3+ is that Unet 3+ implements the full-scale skip connection [3]. Full-scale skip connection enhances the information capture ability in full scales and remain the channel (which is 256 in our Unet 3+) of the feature map in the decoder. 

​	Fig. 2.2 showcases the interconnection between the encoder and $$X_{DE}^2$$ as well as the interconnection between the decoder sub-networks and $$X_{DE}^2$$ . Similar to Unet, $$X_{DE}^2$$  can receive the feature map from same-scale encoder layer $$X_{EN}^2$$ , while the information from low-scale in the encoder $$X_{EN}^1$$ which gains by non-overlapping Max-pooling can also be received by $$X_{DE}^2$$ through the encoder-decoder skip connection, which is differ from Unet. While the interconnection between each neural in decoder can transmit feature map from larger-scale generates by bilinear augmentation in the decoder $$X_{DE}^3$$ and $$X_{DE}^4$$ to $$X_{DE}^2$$. 

<img src=".\res_pic\Unet3+.png" style="zoom:50%;" />

<table>
    <center>
    Figure 2.2 Unet 3+ Structure
    </center>
</table>

### Performance analysis mechanism

​	Totally five mechanisms we chose for analyzing performance: 

​		a).cross entropy

​		b).dice loss

​		c).epoch time

​		d).model size

​		e).noise number

​	Cross entropy is our first choice at the beginning since not only it is a classic loss function of classification problem, but also derivable, which is the determining factor of backpropagation. However, there is no remarkable inter-dependency between cross entropy and dice that is the analysis mechanism on Kaggle, especially when the accuracy of dice rose more than 85%, the further reducing of cross entropy’s loss did not make dice perform better. So the cross entropy was not our main loss function for analysis.

<img src=".\res_pic\negative.png" style="zoom:100%;" />

​	Although dice plays an important role in training and validation, for choosing a well-performance model by calculating the intersection ratio between the overlap area and the average area of ground truth mask and predicted mask, the original dice function transform each probability into separate index and it is not able to satisfy the condition of backpropagation. 

​	Therefore, our group rebuilt the dice function that we called Our_Dice(dice-like) function since the loss function for this course work should be derivable and highly correlative with the evaluation mechanism of Kaggle.

|    **Variable**    |    **Result**    |
| :----------------: | :--------------: |
| Architecture depth |        13        |
|   Loss function    |  Cross entropy   |
|      Dataset       | Original dataset |
|     Optimizer      |       SGD        |
|   Learning rate    |       0.1        |
|      Momentum      |       0.9        |
|   Size of model    |   About 112.4M   |
|      Accuracy      |    About 0.71    |

|    **Variable**    |    **Result**    |
| :----------------: | :--------------: |
| Architecture depth |        4         |
|   Loss function    |  Cross entropy   |
|      Dataset       | Original dataset |
|     Optimizer      |       Adam       |
|   Learning rate    |      0.001       |
|   Size of model    |    About 30M     |
|      Accuracy      |    About 0.84    |

|    **Variable**    |    **Result**    |
| :----------------: | :--------------: |
| Architecture depth |        4         |
|   Loss function    |  Cross entropy   |
|      Dataset       | Original dataset |
|     Optimizer      |       Adam       |
|   Learning rate    |      0.001       |
|   Size of model    |    About 30M     |
|      Accuracy      |    About 0.86    |
