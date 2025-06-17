# 🧼 Handwash Detection using CNN (PyTorch + OpenCV)

This is a computer vision project that detects whether a person is **washing hands** or **not washing hands** using a custom Convolutional Neural Network (CNN). It works on both **images** and **video files**, providing real-time predictions and confidence scores.

---
![Handwash Detection Demo](/output/OUTPUT.mp4)


## 📌 Features

- ✅ Classifies frames as **"Washing Hands"** or **"Not Washing Hands"**
- 📹 Works with **video files** and **static images**
- 🧠 Built with **PyTorch** and **OpenCV**
- 📊 Displays prediction with **confidence percentage**
- 💾 Automatically trains if no saved model is found

---

## 🛠️ Technologies Used

- Python  
- PyTorch  
- OpenCV  
- Pillow (PIL)  
- Torchvision Transforms  

---

## 🎯 Model Details
A simple CNN model trained with Binary Cross Entropy Loss:
- 3 convolutional layers with ReLU + MaxPooling
- Fully connected layers with sigmoid output
- Threshold: 0.6 for classifying as Washing Hands
