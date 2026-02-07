Project Overview: PneumoDeep is a high-precision medical imaging project developed to assist radiologists in the early detection of pneumonia. The system processes raw X-ray data and utilizes a Convolutional Neural Network (CNN) to identify pathological patterns.

Image of a convolutional neural network architecture for image classification
Shutterstock
Technical Workflow:

Data Preprocessing: Implements an automated pipeline for grayscale conversion, resizing to 64×64 pixels, and normalization (pixel values scaled to [0,1]) to ensure model efficiency.

CNN Architecture: Designed a customized Sequential model including multiple Conv2D layers for feature extraction, MaxPooling2D for spatial reduction, and Dropout layers to prevent overfitting.

Categorical Encoding: Utilizes one-hot encoding for medical labels, allowing the model to handle multi-class classification tasks effectively.

Edge Optimization: Features a complete transition from Keras to TensorFlow Lite (TFLite), optimizing the model for deployment on resource-constrained environments like mobile apps or clinical handheld devices.

Advanced Integration: Explores the use of MediaPipe Holistic for potential anatomical landmarking and augmented reality overlays in medical visualization.

Technical Stack:

Core Frameworks: TensorFlow, Keras, MediaPipe.

Data Science: Pandas, NumPy, Scikit-learn.

Computer Vision: OpenCV (cv2).

Optimization: TFLite Converter.
