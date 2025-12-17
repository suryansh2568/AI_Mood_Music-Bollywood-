import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import random

# =========================================================================
# 1. SETUP & CONFIGURATION
# =========================================================================
CLIENT_ID = '5bc83e9abb1842b690f94c11d3eef335'
CLIENT_SECRET = '2a5ad0dfbe044cba93fb6a1cabf58853'

st.set_page_config(page_title="Mood Music Player", page_icon="🎵")
st.title("🎵 Mood Music Player (Strictly Bollywood)")
st.write("Take a selfie. I will play Hindi/Punjabi songs ONLY.")

# =========================================================================
# 2. LOAD MODELS
# =========================================================================
@st.cache_resource
def load_models():
    emotion_model = load_model('emotion_model.hdf5', compile=False)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return emotion_model, face_cascade

try:
    emotion_classifier, face_classifier = load_models()
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    st.success("✅ AI System Ready")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")

# =========================================================================
# 3. SPOTIFY SETUP
# =========================================================================
auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

# =========================================================================
# 4. AGGRESSIVE BOLLYWOOD LOGIC
# =========================================================================
def get_music_link(mood):
    # We use very specific Indian keywords to force Hindi results
    search_query = ""
    
    if mood == 'Happy':
        search_query = "Bollywood Dance Party"  # High energy hindi
    elif mood == 'Sad':
        search_query = "Arijit Singh"           # The king of sad songs (Guaranteed Hindi)
    elif mood == 'Angry':
        search_query = "Divine Rapper"          # Indian Rap/Rock (Gully Boy style)
    elif mood == 'Neutral' or mood == 'Fear':
        search_query = "Bollywood Lo-fi"        # Chill hindi
    elif mood == 'Surprise':
        search_query = "Punjabi Top 50"         # High energy Punjabi
    else:
        search_query = "Bollywood Top 50"

    try:
        # Reduced offset to 5 to keep results relevant (Deep offsets often find junk)
        random_offset = random.randint(0, 5)
        
        # market='IN' is MANDATORY for this to work
        results = sp.search(q=search_query, limit=10, offset=random_offset, type='track', market='IN')
        
        if results and results['tracks']['items']:
            item = random.choice(results['tracks']['items'])
            track_name = item['name']
            artist = item['artists'][0]['name']
            url = item['external_urls']['spotify']
            return track_name, artist, url
    except Exception as e:
        st.error(f"Spotify Error: {e}")
    
    return None, None, None

# =========================================================================
# 5. APP INTERFACE
# =========================================================================
img_file_buffer = st.camera_input("Take a Picture")

if img_file_buffer is not None:
    file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    
    # Process
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Sensitivity: 1.1 for better detection
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # ROI
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (64, 64))
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Predict
            prediction = emotion_classifier.predict(roi, verbose=0)[0]
            label = emotion_labels[prediction.argmax()]
            
            st.header(f"Mood: **{label}**")
            
            # Fetch Song
            track, artist, link = get_music_link(label)
            
            if track:
                st.subheader(f"🎶 {track}")
                st.write(f"🎤 {artist}")
                st.link_button("▶️ Play on Spotify", link)
            else:
                st.warning("No song found.")
            
            # Visuals
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            st.image(frame, channels="BGR")
            break
    else:
        st.warning("No face detected. Move closer!")
