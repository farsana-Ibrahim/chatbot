# AI Chatbot with NLP and Deep Learning - README

An intelligent chatbot built using Natural Language Processing (NLTK) and Deep Learning (Keras) that can understand and respond to user queries through a web interface.

## 🚀 Overview

This project implements a conversational AI chatbot that uses machine learning to understand user inputs and generate appropriate responses. The chatbot is trained on custom intents and patterns, providing a personalized conversational experience.

## ✨ Features

- **Natural Language Understanding**: Processes user messages using NLP techniques
- **Deep Learning Model**: Neural network-based intent classification
- **Web Interface**: Beautiful and responsive chat interface
- **Custom Training**: Train on your own dataset and intents
- **Real-time Responses**: Instant message processing and replies

## 🛠️ Technology Stack

### Backend
- **Python** - Core programming language
- **Flask** - Web framework
- **NLTK** - Natural Language Processing
- **Keras/TensorFlow** - Deep Learning model
- **NumPy** - Numerical computations

### Frontend
- **HTML5/CSS3** - User interface
- **JavaScript** - Client-side functionality
- **jQuery** - AJAX requests and DOM manipulation

### NLP Components
- **WordNet Lemmatizer** - Word normalization
- **Bag-of-Words** - Text representation
- **Tokenization** - Text processing

## 📋 Prerequisites

```bash
# Python 3.7 or higher
# Required system packages
pip install nltk keras tensorflow flask numpy pickle-mixin
```

## 🏗️ Project Structure

```
chatbot-project/
├── app.py                 # Main Flask application
├── training.py            # Model training script
├── model.h5              # Trained neural network model
├── data.json             # Training data and intents
├── texts.pkl             # Vocabulary pickle file
├── labels.pkl            # Classes pickle file
├── templates/
│   └── index.html        # Chat interface
└── static/
    └── styles/
        └── style.css     # Styling for chat interface
```

## 🔧 Installation & Setup

### 1. Install Dependencies
```bash
pip install nltk keras tensorflow flask numpy
```

### 2. Download NLTK Data
```python
import nltk
nltk.download('popular')
nltk.download('punkt')
nltk.download('wordnet')
```

### 3. Prepare Training Data
Create a `data.json` file with your custom intents and patterns:
```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hello", "Hi", "Hey"],
      "responses": ["Hello! How can I help you?", "Hi there!"]
    }
  ]
}
```

### 4. Train the Model
```bash
python training.py
```

### 5. Run the Application
```bash
python app.py
```

Visit `http://localhost:5000` to access the chatbot.

## 🎯 How It Works

### 1. Text Processing Pipeline
- **Tokenization**: Split text into individual words
- **Lemmatization**: Reduce words to their base forms
- **Bag-of-Words**: Convert text to numerical representation

### 2. Intent Classification
- **Neural Network**: 3-layer architecture with dropout
- **Input Layer**: 128 neurons with ReLU activation
- **Hidden Layer**: 64 neurons with ReLU activation
- **Output Layer**: Softmax for intent classification

### 3. Response Generation
- **Pattern Matching**: Find matching intents from training data
- **Random Selection**: Choose from multiple possible responses
- **Confidence Threshold**: Filter predictions below 0.25

## 📊 Model Architecture

```python
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
```

### Training Parameters
- **Optimizer**: SGD with momentum
- **Learning Rate**: 0.01
- **Epochs**: 200
- **Batch Size**: 5
- **Loss Function**: Categorical Crossentropy

## 💬 Chat Interface Features

### User Experience
- **Responsive Design**: Works on desktop and mobile
- **Real-time Messaging**: Instant message delivery
- **Message Bubbles**: Distinct user and bot messages
- **Timestamps**: Message timing information
- **Auto-scroll**: Automatic scrolling to latest messages

### Visual Elements
- **User Avatar**: Customizable user profile image
- **Bot Avatar**: Distinctive chatbot identity
- **Message Styling**: Clean and modern chat bubbles
- **Header**: Professional chatbot branding

## 🔧 Customization

### Adding New Intents
Edit `data.json` to add new conversation topics:
```json
{
  "tag": "weather",
  "patterns": [
    "What's the weather like?",
    "How is the weather today?",
    "Is it raining?"
  ],
  "responses": [
    "I don't have weather information right now.",
    "You might want to check a weather app for that!"
  ]
}
```

### Modifying Model Parameters
Adjust in `training.py`:
- Number of neurons in layers
- Dropout rates
- Learning rate and optimizer
- Training epochs and batch size

### Styling Customization
Modify `style.css` to change:
- Color scheme
- Message bubble styles
- Layout and spacing
- Fonts and typography

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
- Use Gunicorn as WSGI server
- Configure Nginx as reverse proxy
- Set up proper environment variables
- Enable SSL for secure connections

## 📈 Performance

### Model Accuracy
- Trained on custom dataset with multiple intents
- Uses dropout regularization to prevent overfitting
- Implements confidence thresholding for reliable predictions

### Response Time
- Fast inference with pre-trained model
- Efficient text processing pipeline
- Optimized web interface with AJAX

## 🔍 Troubleshooting

### Common Issues

1. **NLTK Data Not Found**
   ```python
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

2. **Model File Missing**
   - Run `training.py` to generate `model.h5`

3. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python version compatibility

4. **Path Issues**
   - Update file paths in `app.py` and `training.py`
   - Use absolute paths for better reliability


