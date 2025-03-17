# 🧠 **Image Classification with Convolutional Neural Networks on CIFAR-100**

## 📂 **Project Overview**  
This project focuses on image classification using the CIFAR-100 dataset, leveraging both a custom convolutional neural network (CNN) and pre-trained models with PyTorch. The objective was to build a robust model by comparing a custom-built CNN architecture with popular pre-trained backbones, specifically ResNet-18.

## 🎯 **Objectives**  
- Develop a custom convolutional neural network (CNN) with a clear separation between the **backbone** and **classifier** components.  
- Evaluate the performance of the custom model against pre-trained backbones, including **ResNet-18**.  
- Analyze and compare the results, focusing on accuracy, training time, and model complexity.

## 🛠 **Tech Stack**  
- **Programming Language:** Python  
- **Libraries:** PyTorch, torchvision, NumPy, Matplotlib  
- **Dataset:** CIFAR-100

## 🚀 **Approach**  
1. **Data Preparation:** Loaded and preprocessed the CIFAR-100 dataset using PyTorch's `torchvision.datasets`.  
2. **Custom Model Development:** Implemented a custom CNN with distinct backbone and classifier modules.  
3. **Pre-trained Model Comparison:** Utilized the **ResNet-18** architecture from `torchvision.models` as a pre-trained backbone for comparison.  
4. **Training and Evaluation:** Trained both models, recorded performance metrics, and visualized the results.  
5. **Analysis:** Compared the custom and pre-trained models based on accuracy, learning curves, and inference speed.

## 📊 **Results** 
- All results can be found in the CIFAR100_Report file, avaliable as both .pdf and .ipynb.
- The pre-trained **ResNet-18** backbone achieved higher accuracy and faster convergence compared to the custom CNN.  
- The custom CNN provided valuable insights into model design and feature extraction but required more epochs to achieve competitive performance.
- In the results folder you can find .csv and .xlsx tables containing all trained and tested models with their results and short description.
