# Enginering Thesis - Human Posture Detection
This project focused on detecting human posture using stereovision techniques. A dual-camera system was used to capture synchronized images and perform 3D triangulation based on 2D keypoints from pose estimation. A dataset of synchronized 2D and 3D coordinates was created to train and evaluate recurrent neural networks. Three model types—RNN, GRU, and LSTM—were tested with varying parameters to compare their effectiveness. The LSTM model achieved the highest validation accuracy, demonstrating its strong potential for 3D posture prediction from stereoscopic video data.

## Thesis introduction

The first chapter of the thesis introduces the primary goal of developing a human posture detection system using stereovision and deep learning techniques. It highlights the importance of posture analysis in fields like healthcare, ergonomics, and sports science, while addressing the challenges of accurate 3D posture detection. The chapter includes a literature review on various methods of pose estimation, including traditional motion capture systems, deep convolutional neural networks, and the use of recurrent neural networks (RNNs) for temporal analysis. It also examines the advantages of stereovision over other 3D reconstruction methods like LiDAR or Kinect, emphasizing its potential for low-cost, accurate results. Finally, the chapter outlines the project’s objectives, including building a dual-camera system for 2D keypoint extraction, converting these points into 3D coordinates, and training an RNN for accurate 3D posture prediction.

**Mathematical representation:**

Points in 2D space are represented as \( u \) and \( v \), corresponding to the x and y coordinates on the image:

$$
m = \begin{bmatrix} u \\ v \end{bmatrix}
\qquad \widetilde{m} = \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
$$

Points in 3D space are represented as:

$$
M = \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}
\qquad
\widetilde{M} = \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
$$

Camera intrinsic parameter matrix:

$$
K = \begin{bmatrix}
f_x & s & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
$$

- \( c_x \), \( c_y \) – principal point coordinates,
- \( f_x \), \( f_y \) – scaling factors,
- \( s \) – parameter describing angular distortion of the two image axes.

Rotation matrix around the x-axis, representing a rotation by angle \( \alpha \):

$$
R_x(\alpha) = 
\begin{bmatrix}
1 & 0 & 0 \\
0 & \cos(\alpha) & -\sin(\alpha) \\
0 & \sin(\alpha) & \cos(\alpha)
\end{bmatrix}
$$

Rotation matrix around the y-axis, representing a rotation by angle \( \beta \):

$$
R_y(\beta) = 
\begin{bmatrix}
\cos(\beta) & 0 & \sin(\beta) \\
0 & 1 & 0 \\
-\sin(\beta) & 0 & \cos(\beta)
\end{bmatrix}
$$

Rotation matrix around the z-axis, representing a rotation by angle \( \gamma \):

$$
R_z(\gamma) = 
\begin{bmatrix}
\cos(\gamma) & -\sin(\gamma) & 0 \\
\sin(\gamma) & \cos(\gamma) & 0 \\ 
0 & 0 & 1
\end{bmatrix}
$$

Combined rotation matrix:

$$
R = R_z(\gamma) R_y(\beta) R_x(\alpha)
$$

Camera translation vector, describing its position:

$$
\mathbf{t} =
\begin{bmatrix}
t_1 \\
t_2 \\
t_3
\end{bmatrix}
$$

- \( t_1 \) – camera translation along the x-axis,
- \( t_2 \) – camera translation along the y-axis,
- \( t_3 \) – camera translation along the z-axis.

Camera extrinsic parameter matrix:

$$
[R \,|\, t] =
\begin{bmatrix}
R_{11} & R_{12} & R_{13} & t_1 \\
R_{21} & R_{22} & R_{23} & t_2 \\
R_{31} & R_{32} & R_{33} & t_3
\end{bmatrix}
$$

The relationship between 2D and 3D points:

$$
\lambda \widetilde{m} = K \begin{bmatrix} R & t \end{bmatrix} M
$$

- \( \lambda \) – scaling factor,
- \( R \), \( t \) – camera extrinsic parameters, representing the position and orientation in space.

## Calibration and synchronization of the cameras
This chapter focuses on the calibration and synchronization of the cameras, which are essential for accurate 3D human posture detection. It explains the process of calibrating both cameras to eliminate lens distortions and align their intrinsic and extrinsic parameters. Synchronization of the cameras is discussed, ensuring both capture frames simultaneously to maintain the precision of depth estimation. The chapter also describes the data collection process, where images are captured under various conditions to build a diverse dataset for pose estimation and 3D reconstruction. Overall, it emphasizes the importance of proper calibration, synchronization, and data collection for the accuracy and reliability of the system.

<p align="center">
  <img src="https://github.com/user-attachments/assets/2dc585fc-c475-49bc-b68a-1c0462c7e80f" alt="image">
</p>

## Data Preparation and Processing
The chapter on "Data Preparation and Processing" discusses the essential steps taken to collect, organize, and preprocess the data required for the human posture detection system. Initially, the process of camera calibration and synchronization is outlined, ensuring that the dual-camera setup captures synchronized video streams with precise alignment. The chapter then describes the procedure for extracting 2D keypoints from the synchronized video feeds using a pre-trained pose estimation model. It goes on to explain the transformation of these 2D points into 3D coordinates using stereovision techniques such as triangulation. Furthermore, data augmentation methods and normalization techniques are employed to ensure that the neural network receives diverse, high-quality input, helping to improve the model's performance and robustness for predicting human posture over time.

<p align="center">
  <img src="https://github.com/user-attachments/assets/46fd2fed-be18-4d6b-9f3c-39128c49296a" alt="image">
</p>

## Training Activity Detection Models
The chapter on "Training Activity Detection Models" focuses on the development and training of machine learning models to detect and predict human activities based on posture data. It begins by discussing the selection of appropriate deep learning architectures, with a particular emphasis on recurrent neural networks (RNNs), including Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU), for their ability to capture temporal dependencies in human movement. The chapter details the process of training the models using labeled datasets of human poses, where the 3D coordinates obtained from the stereovision system are used as input. It also addresses the challenges encountered during training, such as overfitting, and outlines the techniques used to mitigate these issues, including dropout and early stopping. Finally, the chapter evaluates the performance of the trained models, presenting results on their ability to accurately classify and predict human activities in real-time, based on the 3D posture data.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d81db2a2-f285-43c4-84f8-d2f9928e8883" alt="image">
</p>

## Application
The "Application" chapter focuses on the practical implementation and deployment of the human posture detection system. It begins with the design of a user-friendly interface, where real-time posture tracking and activity detection are displayed to the user, providing immediate feedback on their movements. The chapter discusses the integration of the previously trained models into a functional application, detailing the steps taken to ensure that the system operates efficiently with live data from the dual-camera setup. It also covers the challenges of optimizing the application for speed and accuracy, ensuring smooth synchronization of the cameras and reliable processing of the 3D pose data. Finally, the chapter highlights the application’s potential use cases, including healthcare, fitness tracking, and ergonomic assessments, emphasizing how it can be employed in real-world scenarios to monitor and improve posture and movement.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f67935bb-1924-415c-ae2e-7ef202fd4c69" alt="image">
</p>

## Results
The goal of this engineering project was to develop a system for human activity recognition using stereovision technology. A stereo camera setup was calibrated and synchronized, enabling the use of the BlazePose model to extract 3D body keypoints through triangulation. The collected 3D trajectories were used to train recurrent neural networks (Simple RNN, GRU, LSTM) for classifying five types of actions, achieving over 90% accuracy on validation data. A lightweight application was built using Gradio, allowing users to perform calibration and activity recognition without technical knowledge. The results confirm that accurate and efficient human activity recognition is feasible with low hardware requirements and offers potential for further real-world and commercial applications.




