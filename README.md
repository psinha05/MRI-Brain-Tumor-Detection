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
VGG16, which is a pre-trained convolutional neural network (CNN) for image classification.

First, the VGG16 model is loaded with input_shape=(IMAGE_SIZE,IMAGE_SIZE,3), include_top=False, weights='imagenet'. The input shape is set to match the size of the images in the dataset, which is 128x128 pixels. The include_top parameter is set to False, which means that the final fully-connected layers of VGG16 that perform the classification will not be included. The weights parameter is set to 'imagenet' which means that the model will be pre-trained with a dataset of 1.4 million images called imagenet.


Next, the for layer in base_model.layers: loop is used to set all layers of the base_model (VGG16) to non-trainable, so that the weights of these layers will not be updated during training.

Then, the last three layers of the VGG16 model are set to trainable by using base_model.layers[-2].trainable = True,base_model.layers[-3].trainable = True and base_model.layers[-4].trainable = True

After that, a Sequential model is created and the VGG16 model is added to it with model.add(base_model).


Next, a Flatten layer is added to the model with model.add(Flatten()) which reshapes the output of the VGG16 model from a 3D tensor to a 1D tensor, so that it can be processed by the next layers of the model.

Then, a Dropout layer is added with model.add(Dropout(0.3)) which is used to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training time.


After that, a dense layer is added with 128 neurons and relu activation function is added with model.add(Dense(128, activation='relu')).


Next, another Dropout layer is added with model.add(Dropout(0.2))


Finally, the output dense layer is added with number of neurons equal to the number of unique labels and 'softmax' activation function is added with model.add(Dense(len(unique_labels), activation='softmax')). The 'softmax' activation function is used to give a probability distribution over the possible classes.


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


