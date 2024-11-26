# Deep Learning Lab: Image Classification

## Objective
The primary goal of this lab was to develop a deeper understanding of the PyTorch library by constructing and evaluating various neural network architectures. This included implementing models such as Convolutional Neural Networks (CNNs), Faster R-CNN, and Vision Transformers (ViT) for the task of classifying images from the MNIST dataset.

---

## Summary of Work

### Part 1: CNN and Faster R-CNN
1. **CNN Classifier**:
   - Dataset: MNIST (28x28 grayscale images).
   - Architecture:
     - Built with layers including convolution, pooling, and fully connected layers.
     - Incorporated ReLU activation, dropout for regularization, and softmax for classification.
     - Training performed on GPU for faster computation.
   - Hyperparameters:
     - Optimizer: Adam
     - Loss Function: CrossEntropyLoss
     - Kernels, padding, and learning rate tuned for optimal performance.

2. **Faster R-CNN**:
   - Adapted for MNIST classification.
   - Applied object detection principles for image classification.
   - Achieved strong performance but required longer training time compared to CNN.

3. **Comparison**:
   - **CNN**:
     - Accuracy: 98.88%
     - F1 Score: 0.9888
     - Training Time: 76.59 seconds
   - **Faster R-CNN**:
     - Accuracy: 99%
     - F1 Score: 0.9902
     - Training Time: 122.71 seconds
   - **Observations**: Faster R-CNN achieved marginally better performance but at the cost of longer training time.

---

### Part 2: Vision Transformer (ViT)
1. **Implementation**:
   - A Vision Transformer architecture was constructed following a tutorial.
   - Images were processed into token sequences and fed into transformer layers for classification.
   - Used positional encodings and self-attention mechanisms to process the MNIST dataset.

2. **Comparison**:
   - ViT demonstrated strong performance and its capability to capture global image features.
   - However, it required higher computational resources and longer training time compared to CNN and Faster R-CNN.

---

## Key Learnings
- CNNs are highly efficient for simple image classification tasks, providing a good balance between performance and resource usage.
- Faster R-CNN, while designed for object detection, also performs well for classification but with a trade-off in training time.
- Vision Transformers (ViT) highlight the potential of transformer-based architectures for computer vision, excelling in global feature extraction but demanding greater computational power.

---

## Conclusion
This lab highlighted the trade-offs between various neural architectures for image classification. CNNs and Faster R-CNNs are effective for simpler tasks, while ViT showcased its potential for future work on more complex datasets.
