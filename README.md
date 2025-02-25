# AI-Powered Defect Detection in Manufacturing of O-rings

### Project Overview
This project focuses on developing an **AI-powered defect detection** system for O-rings, a critical component in industries such as automotive, aerospace, and healthcare. The system leverages deep learning models like **_ResNet_** for binary classification and **_Autoencoders_** for anomaly detection to identify microscopic defects such as cracks, tears, and dimensional inconsistencies. By automating defect detection, the project aims to enhance quality control, reduce production costs, and improve operational efficiency.

### Installation and Setup
#### Prerequisites
Ensure you have the following installed on your system:
- Python (>= 3.8)
- Node.js (>= 16.x) and npm (>= 8.x)
- Virtual Environment (venv or equivalent)

### Backend Setup (Python)
1. Clone the Repository:
```
git clone https://github.com/BipinRajC/O-Ring-Defect-Detection.git  
cd O-Ring-Defect-Detection
```

2. Create a Virtual Environment:
```
python -m venv venv
```

3. Activate the Virtual Environment:
- On Windows:
```
venv\Scripts\activate
```
- On macOS/Linux:
```
source venv/bin/activate
```

4. Install Required Packages:
```
pip install -r requirements.txt
```

5. Run the Backend:
```
streamlit run main.py
```

### Frontend Setup (Typescript)
1. Navigate to the Frontend Folder:
```
cd frontend/
```

2. Install Dependencies:
```
npm install
```

3. Run the frontend:
```
npm run dev
```

### How to use
- Start the backend server using the streamlit run app.py command.
- Navigate to the frontend folder and run the frontend using npm run dev.
- Open the provided URL in your browser to access the user interface.
- Upload images of O-rings through the interface to detect defects in real-time.
- View detailed logs, including defect type, location, and severity.

### Methodology
The project employs a combination of deep learning models and a threshold-based ensemble method for defect detection:

- ResNet-18: A convolutional neural network (CNN) used for binary classification of O-rings into defective and non-defective categories.
- Autoencoders: Used for anomaly detection by identifying deviations from normal, defect-free O-rings.
- Threshold-Based Ensemble Logic: Combines predictions from ResNet and autoencoders to improve accuracy and reliability.

The system is optimized for real-time defect detection, ensuring rapid processing speeds suitable for manufacturing environments. A user-friendly interface built with TypeScript provides operators with real-time feedback, defect heatmaps, and detailed logs.

### Results
The AI-powered defect detection system achieved the following:

- Accuracy: 93% overall accuracy in detecting defects.
- Real-Time Processing: Optimized for live manufacturing environments.
- Improved Quality Control: Reduced human error and increased defect detection reliability.

### Tools and Techniques

- Backend: Streamlit for managing API calls and processing image data.
- Frontend: TypeScript for building an interactive and user-friendly interface.
- Deep Learning Models:
ResNet-18 for binary classification + Autoencoders for anomaly detection.
- Image Processing: OpenCV for preprocessing tasks like resizing, normalization, and augmentation




