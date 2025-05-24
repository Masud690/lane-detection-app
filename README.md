# lane-detection-

# 🚗 Lane Detection Web App using Streamlit

This is a simple web application that detects and highlights road lane lines in uploaded videos using **OpenCV** and **Streamlit**. It is designed to demonstrate basic computer vision techniques for lane detection — useful for self-driving car simulations or educational purposes.

---

## 📸 Features

- Upload your own road videos (`.mp4` or `.avi`)
- Detects straight lane lines using:
  - Grayscale conversion
  - Gaussian blur
  - Canny edge detection
  - Region of interest (masking)
  - Hough Line Transform
- Streams the processed video frame by frame
- Option to download the output video
