# MRI-Brain-Tumor-Detection : 
Brain tumors are classified as primary (originating in the brain) or secondary (metas-tasizing from elsewhere). Gliomas are the most prevalent malignant primary brain tumor in adults, accounting for 80%.According to the WHO, gliomas are classified into four grades: low grade (LGG) (grades 1-2), which are less prevalent and have low blood concentration and sluggish growth, and high grade (HGG) (grades 3-4), which have rapid growth and aggressiveness.

In This application we have using the BRATS-2020 dataset, Early diagnosis of brain tumor can help one avoiding the painful consequences as well as the life risk. Eventually, this diagnosis is the most crucial part since manual delineation of tumor form brain MRI is quite time consuminig and laborious also. A number of advanced Magnetic Resonance Imaging (MRI) techniques that include Diffusion Tensor Imaging (DTI), MR Spectroscopy (MRS) and Perfusion MR are used for the analysis of brain tumor through MRI.

Deep learning approaches have achieved state-of-the-art performance for automatic brain tumor segmentation using multi-model MRIs. CNN is a highly effective method for picture ecognition and prediction. CNN is predominantly employed for brain tumor seg-mentation, classification, and survival time prediction. 

BRATS 2020 Dataset BRATS 2020 is a recently published benchmark dataset for the purpose of segment-ing the brain tumor.Indeed, a number of studies have already used this dataset for reporting and comparing their results. This 3D multimodal MRI dataset consists of 369 training cases, 125 validation cases 166 test cases. Among the training cases, 293 cases include the HGG tumor and 76 cases include the LGG tumor.

MRI Brain Tumor Detection using the BRATS 2020 dataset with a VGG16-based architecture is an approach for classifying brain tumor types from MRI images. VGG16, a Convolutional Neural Network (CNN) model known for its simplicity and depth, is often utilized in image classification tasks.

Overview of the Process

Data Preprocessing:
BRATS Dataset: The BRATS 2020 dataset is a collection of multimodal MRI scans (Flair, T1, T1c, T2) with labeled data indicating the presence and types of brain tumors. The goal is to detect and classify tumors from MRI scans, typically into categories like "Glioma," "Meningioma," or "No Tumor."

Preprocessing Steps:
Resizing the input images to fit the model (typically 128x128 pixels for VGG16).Normalization of pixel values to a range (e.g., 0 to 1) by dividing by 255.
Data augmentation (if necessary) to improve model generalization, like rotation, scaling, or flipping.

Model Architecture (using VGG16):
VGG16 is used as a base model in the sequential architecture. It consists of a series of convolutional layers followed by max-pooling layers. Here’s the flow and structure of VGG16:
VGG16 Layers:

Convolutional Layers:
These layers apply filters to input images to detect patterns like edges, textures, and shapes. VGG16 uses a stack of convolution layers with 3x3 kernels and strides of 1.
Fully Connected Layers (Dense Layers): After the convolutional layers, the 3D feature maps are flattened into 1D vectors and passed through fully connected layers:
Dense(output size, e.g., 3 for three classes in the tumor classification).
These dense layers use ReLU activation functions, except the final output layer.

Output Layer: The output layer uses Softmax activation for multi-class classification. Softmax assigns probabilities to each class and the class with the highest probability is selected as the output. This is crucial for distinguishing between different tumor types (or identifying no tumor).

Sequence Model Design:
Input Layer: This layer is typically a 128x128 image (RGB channels) that is fed into the VGG16 architecture.
Base VGG16 Model: The pretrained VGG16 is used as the base model, where the convolutional and pooling layers extract high-level features.
Additional Dense Layers: Once the features are extracted, a few additional fully connected layers are added for classification:
A Dense layer with ReLU activation.
Another Dense layer with ReLU activation.
Output Dense layer with Softmax for the final classification (e.g., “No Tumor,” “Glioma,” or “Meningioma”).

The final layer is a softmax layer with a number of units corresponding to the number of classes you want to classify. In tumor detection, this is usually 3: no tumor, glioma, or meningioma.
Softmax Activation: This layer converts the output to probabilities for each class.

Key Points:


Pre-trained Weights: VGG16 is typically used with weights pre-trained on ImageNet, which helps in faster convergence and better performance for image classification tasks.
Fine-tuning: we have choose to fine-tune some of the layers of VGG16 by unfreezing (2nd, 3rd, 4th last layer) the base layers after training the model with the frozen base.
Dropout: Dropout layers can be added between fully connected layers to prevent overfitting (Dropout upto 20%)
Optimizer: Adam optimizer is commonly used, as it adapts the learning rate during training.

Training the Model:

Train the model using the prepared MRI dataset, specifying a loss function (categorical cross-entropy for multi-class classification), and an optimizer (like Adam).
During training, the model learns to map input MRI images to their corresponding tumor type classifications.

Evaluation:

Once the model is trained, evaluate it using test data to check accuracy, precision, recall, and F1-score.
Consider visualizing the results using confusion matrices or ROC curves to understand the model's performance better.

Conclusion:

Using VGG16 for brain tumor detection is a powerful way to leverage deep learning for medical image classification. By utilizing transfer learning with the pre-trained weights from ImageNet, the model can generalize well, even for a specialized dataset like BRATS. The final softmax output allows the model to classify MRI images into different categories based on the type of tumor detected.


