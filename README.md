---
title: Cattle Breed Classifier
emoji: 🐄
colorFrom: blue
colorTo: green
sdk: docker
app_file: Dockerfile
pinned: false
---

# Cattle Breed Classifier

An AI-powered web application for classifying indigenous Indian cattle breeds using deep learning. Supports 26 breeds including cows and buffalos.

## Features

- **Real-time Classification**: Upload images or capture from camera
- **26 Indian Breeds**: Comprehensive coverage of indigenous cattle
- **High Accuracy**: ResNet50 model trained on 3,000+ images
- **Breed Information**: Detailed info on milk yield, primary use, and characteristics
- **Responsive Design**: Works on desktop and mobile

## Tech Stack

- **Backend**: FastAPI + PyTorch
- **Frontend**: React + Vite
- **Model**: ResNet50 Transfer Learning
- **Deployment**: Docker + Hugging Face Spaces

## Usage

1. Upload an image of a cattle
2. Get instant breed prediction with confidence score
3. View detailed breed information
4. Explore all supported breeds in the browser

## Dataset

- 3,056 images across 26 breeds
- 21 cow breeds + 5 buffalo breeds
- Images resized to 224×224 pixels
- Stratified train/val/test split (70/15/15)

## Model Performance

- **Accuracy**: 85%+
- **F1 Score**: 0.82
- **Inference Time**: <100ms

## Supported Breeds

**Cows**: Alambadi, Amritmahal, Bargur, Dangi, Deoni, Gir, Hallikar, Kangayam, Kankrej, Kasaragod, Kenkatha, Kherigarh, Malnad Gidda, Nagori, Nagpuri, Nimari, Pulikulam, Rathi, Sahiwal, Tharparkar, Umblachery

**Buffalos**: Banni, Jaffrabadi, Mehsana, Nili Ravi, Shurti

## Development

```bash
# Install dependencies
pip install -r requirements.txt
npm install

# Run locally
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

## License

MIT License
