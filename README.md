# Audio CNN

## Overview

AudioClassifier is a full-stack application for real-time environmental sound classification, featuring a custom ResNet-style deep CNN that converts audio into Mel Spectrograms for robust feature extraction and classification. The backend, built with Python, PyTorch, and FastAPI, is optimized for scalable, serverless GPU inference using Modal, while the interactive frontend - developed with Next.js, React, and Tailwind CSS - enables users to upload audio, view predictions with confidence scores, and visualize internal model feature maps, waveforms, and spectrograms, making it a comprehensive platform for both audio AI

## Features:

- 🧠 Deep Audio CNN for sound classification
- 🧱 ResNet-style architecture with residual blocks
- 🎼 Mel Spectrogram audio-to-image conversion
- 🎛️ Data augmentation with Mixup & Time/Frequency Masking
- ⚡ Serverless GPU inference with Modal
- 📊 Interactive Next.js & React dashboard
- 👁️ Visualization of internal CNN feature maps
- 📈 Real-time audio classification with confidence scores
- 🌊 Waveform and Spectrogram visualization
- 🚀 FastAPI inference endpoint
- ⚙️ Optimized training with AdamW & OneCycleLR scheduler
- 📈 TensorBoard integration for training analysis
- 🛡️ Batch Normalization for stable & fast training
- 🎨 Modern UI with Tailwind CSS & Shadcn UI
- ✅ Pydantic data validation for robust API requests

## Setup

Follow these steps to install and set up the project.

### Clone the Repository

```bash
git clone https://github.com/sid995/AudioClassifier.git
```

### Install Python

Download and install Python if not already installed. Use the link below for guidance on installation:
[Python Download](https://www.python.org/downloads/)

Create a virtual environment with **Python 3.12**.

### Backend

Navigate to folder:

```bash
cd AudioClassifier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Modal setup:

```bash
modal setup
```

Run on Modal:

```bash
modal run main.py
```

Deploy backend:

```bash
modal deploy main.py
```

### Frontend

Install dependencies:

```bash
cd frontend
npm i
```

Run:

```bash
npm run dev
```
