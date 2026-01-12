
# FDAA Project 2025 - Final Exam

The objective of this topic is to assess your knowledge in designing a neural network architecture capable of performing image classification.  
The images relate to sign language. Each image contains a hand sign representing a digit between 0 and 9. Thus, the goal is similar to what you have already achieved with the MNIST image classification task.

All Keras codes you have developed during the sessions are allowed during the exam. At the end of the test, you will send or share your "exam notebook" with me at: sebastien.ambellouis@gmail.com.  
This evaluation consists of several questions that outline the steps to achieve the expected model. Therefore, you must submit a clear and easily usable notebook.  
Note: Distribute your code into multiple **successive code cells**. I also ask you to annotate and comment on your notebook thoroughly using **text cells** to demonstrate your understanding of the operations you perform.

My grading is straightforward: I execute each cell of your notebook to check for errors and verify whether the expected results are obtained.  
If I need to correct any errors, penalties will apply. Answer the questions in order to progress smoothly and accumulate points.

## Good luck and have fun!

The image dataset and their annotations are provided in the two numpy files: **X.npy** and **Y.npy**.

### Information about the image dataset (**X.npy**):
- Image resolution: 64x64  
- Color space: Grayscale  
- Number of classes: 10 (Digits from 0 to 9)  

### Information about the annotations (**Y.npy**):
- One label per row  
- Each label is encoded using the "one-hot" technique, which you know encodes each of the 10 classes on 10 bits with one bit set to 1 for a given class.  

The signs are as follows (0 to 4 on the first row, and 5 to 9 on the second row):  

![image](signdataset.png)  

In the following tasks, you must define, train, evaluate, and infer the following neural network architecture:

## Functional Model  
<img src="functional_model.png" alt="Functional Model" width="400"/>

As you can observe, this architecture cannot be defined sequentially because the image input feeds two parallel branches.  
(See [https://github.com/SebAmb/FDAA2025/blob/main/sequential%20vs%20functional.md]())

---

### **Question 1**  
Load the two numpy files into two variables. Display the "shape" of these two variables and show image 0 and image 2000 along with their respective annotations.

### **Question 2**  
Check the maximum and minimum grayscale values across all images. Normalize each image if necessary.

### **Question 3**  
Create a test subset from the dataset of images you loaded and potentially normalized earlier. The test dataset should contain 15% of the images from the full dataset.

### **Question 4.1**  
In this question, define only the left branch of the neural network model presented in the previous diagram. Use the functional Keras API description to define it.  
(See [ttps://github.com/SebAmb/FDAA2025/blob/main/sequential%20vs%20functional.md](https://github.com/SebAmb/FDAA2025/blob/main/sequential%20vs%20functional.md)).  

- The filters are of size 3x3.  
- The activation function for all layers is ReLU.  

Make sure to define the shape of your input according to the image characteristics.

### **Question 4.2**  
Create the final model by modifying the model from **Question 4.1** to include the second branch, also known as the residual branch.  

### **Question 5**  
In this question, you will train the model described in **Question 4.2**.  
To do this, follow these steps:  
- Compile the model.  
- Define the loss function and accuracy metric.  
- Set the number of epochs (`epochs=20`).  
- Set the batch size.  

Training should be performed on a training dataset and a validation dataset at each epoch.  

Add a callback function to save the network's weights whenever they produce better results than previous weights. This will allow you to retain the "best" learned weights for future use.

### **Question 6.1**  
Once training is completed (number of epochs = 20), display the learning curves to assess any overfitting phenomenon.

### **Question 6.2**  
Evaluate your best model on the test dataset.  
What accuracy is achieved?  
Display the confusion matrix for this model.

### **Question 7**  
In a new cell of your notebook, create a new model by adding a BatchNormalization layer after each convolutional layer.
Restart the training process and plot the learning curves.
Compare the obtained performance with the results from Question 5.

IMPORTANT: Do not directly modify the cells from **4.1**, **4.2**, **5**, **6.1**, and **6.2**. Use new code cells instead.

### **Question 8**  
Evaluate your best model on the test dataset.  
What accuracy is achieved?  
Display the confusion matrix for this model.

### **Question 9**  
Infer the model on the images found in the evaluation folder on GitHub:  
[https://github.com/SebAmb/FDAA2025/tree/main/evaluation](https://github.com/SebAmb/FDAA2025/tree/main/evaluation)  

The objective is to verify that the model can recognize the hand signs it has learned from the images in the folder.

--- 
