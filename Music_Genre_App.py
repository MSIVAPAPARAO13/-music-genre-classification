import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize
from tensorflow.keras.models import load_model

#Function
@st.cache_resource
def load_trained_model():
    model = tf.keras.models.load_model('Hybrid_CRNN_Model.keras', compile=False)

    return model

# Load and preprocess audio data
def load_and_preprocess_file(file_path, target_shape=(150, 150)):
    data = []
    file_path.seek(0)
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    # Perform preprocessing (e.g., convert to Mel spectrogram and resize)
    # Define the duration of each chunk and overlap
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds

    # Convert durations to samples
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate

    # Calculate the number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    # Iterate over each chunk
    for i in range(num_chunks):
                    # Calculate start and end indices of the chunk
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        
                    # Extract the chunk of audio
        chunk = audio_data[start:end]
        if len(chunk) == 0:
            continue

                    # Compute the Mel spectrogram for the chunk
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
                #mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

    return np.array(data)

def model_prediction(X_test):
    model = load_trained_model()
    # Get raw probabilities for all chunks
    probabilities = model.predict(X_test, verbose=0)
    # Calculate mean confidence across all chunks
    mean_probabilities = np.mean(probabilities, axis=0)
    # Final class selection
    predicted_idx = np.argmax(mean_probabilities)
    return predicted_idx

## Streamlit UI
st.sidebar.title("Dashboard")

app_mode = st.sidebar.selectbox("Select Page",["Home", "About", "Prediction"] )

# Main Page
if( app_mode == "Home"):
    st.markdown(''' ## Welcome to the \n
    ## Music Genre Classification System !''')
    image_path = "Images/home.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown('''
    **Our goal is to help in identifying music genres from audio tracks efficiently. Upload an audio file, and our system will analyze it to detect its genre. Discover the power of AI in music analysis!**

### How It Works
1. **Upload Audio:** Go to the **Genre Classification** page and upload an audio file.
2. **Analysis:** Our system will process the audio using advanced algorithms to classify it into one of the predefined genres.
3. **Results:** View the predicted genre along with related information.

### Why Choose Us?
- **Accuracy:** Our system leverages state-of-the-art deep learning models for accurate genre prediction.
- **User-Friendly:** Simple and intuitive interface for a smooth user experience.
- **Fast and Efficient:** Get results quickly, enabling faster music categorization and exploration.

### Get Started
Click on the **Genre Classification** page in the sidebar to upload an audio file and explore the magic of our Music Genre Classification System!

### About Us
Learn more about the project, our team, and our mission on the **About** page.
    ''')

elif app_mode == "About":
    st.markdown("""
## 🎵 About the Project

Music has always been a fascinating subject for analysis. Researchers have long tried to understand:

- What differentiates one song from another?
- How can sound be visualized?
- What makes tones unique?

This project explores these questions using machine learning and audio data.

---

## About the Dataset

### Content Overview

**1. Genres (Original Audio)**  
A collection of 10 genres with 100 audio files each.  
Each audio clip is 30 seconds long.  
This is the well-known **GTZAN dataset** (often called the *MNIST of audio*).

**2. List of Genres**
- Blues  
- Classical  
- Country  
- Disco  
- Hip-hop  
- Jazz  
- Metal  
- Pop  
- Reggae  
- Rock  
""")



    #Prediction Page
elif app_mode == "Prediction":
    st.header("🎧 Model Prediction")

    test_mp3 = st.file_uploader(
        "Upload an Audio File", type=["mp3"]
    )

    if test_mp3 is not None:
        st.audio(test_mp3)

        if st.button("Predict Genre"):
            with st.spinner("Processing..."):

                X_test = load_and_preprocess_file(test_mp3)
                if len(X_test) == 0:
                    st.error("Audio processing failed. Try another file.")
                else:
                    result_index = model_prediction(X_test)

                    labels = [
                        'blues', 'classical','country','disco',
                        'hiphop','jazz','metal','pop','reggae','rock'
                    ]

                    st.success(f"🎶 Predicted Genre: {labels[result_index]}")