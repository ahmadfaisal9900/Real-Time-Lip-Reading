# LipNet: Real-Time Lip Reading

This repository implements a real-time lip reading system using deep learning. The system consists of two main components:
- **Training Notebook** (`LipNet Training.ipynb`): A Jupyter Notebook for training a LipNet-based model on lip reading datasets.
- **Real-Time Inference** (`RealTimeInference.py`): A Python script that uses OpenCV and TensorFlow to perform real-time lip reading from webcam input.

## Features
- **Convolutional + LSTM Network**: Uses a combination of 3D CNNs and Bidirectional LSTMs to capture spatial and temporal lip movement patterns.
- **CTC Loss for Sequence Prediction**: Implements Connectionist Temporal Classification (CTC) for recognizing spoken words from a sequence of lip movements.
- **Real-Time Inference**: Processes live video input from a webcam, isolates the mouth region, and predicts spoken text in real time.
- **Pre-trained Model Integration**: Loads a pre-trained model for immediate inference without requiring retraining.

## Installation
To use this project, install the necessary dependencies:

```bash
pip install tensorflow opencv-python numpy
```

## Usage

### Training
1. Open `LipNet Training.ipynb` in Jupyter Notebook.
2. Follow the steps to train the model on a lip reading dataset.
3. Save the trained model for inference.

### Real-Time Inference
1. Ensure a trained model checkpoint is available.
2. Run the real-time inference script:

```bash
python RealTimeInference.py
```

3. The script will:
   - Capture video from the webcam.
   - Preprocess each frame to isolate the mouth region.
   - Perform lip reading using the trained model.
   - Display the predicted text on the screen.

Press `q` to exit the real-time inference.

## Model Architecture
- **Conv3D Layers**: Extract spatial features from lip movement frames.
- **Bidirectional LSTMs**: Learn temporal dependencies in speech patterns.
- **CTC Loss**: Enables sequence-to-sequence learning without requiring frame-level labels.

## Example Output
When running real-time inference, you may see output like:

```
Original: hello world
Prediction: helo w0rld
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

## Future Improvements
- Train on larger datasets for improved accuracy.
- Optimize real-time processing speed.
- Implement word-level language modeling to improve predictions.

## Credits
This implementation is inspired by the **LipNet** model and builds upon modern deep learning techniques for automatic speech recognition from video.

