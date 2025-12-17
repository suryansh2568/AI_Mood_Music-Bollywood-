import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import webbrowser
import time
import random

# 1) Configuration
CLIENT_ID = '5bc83e9abb1842b690f94c11d3eef335'
CLIENT_SECRET = '2a5ad0dfbe044cba93fb6a1cabf58853'
REDIRECT_URI = 'http://127.0.0.1:5000/callback' 

# 2) Setup Spotify and Ai models
print("Booting up... connecting to Spotify...")

try:
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope="user-library-read playlist-modify-public"
    ))
    print("Spotify Connected!")
except Exception as e:
    print(f"Spotify Auth Failed: {e}")

print("Loading AI Brain...")
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotion_model_path = 'emotion_model.hdf5'  
try:
    emotion_classifier = load_model(emotion_model_path, compile=False)
    print("AI Model Loaded!")
except OSError:
    print("ERROR: 'emotion_model.hdf5' not found.")
    exit()

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 3) Helper
def play_mood_music(mood):
    print(f"\n[INFO] Detected Mood: {mood}")
    print(f"[INFO] Searching Spotify directly for {mood} songs...")
    
    # 1. Define Search Terms
    search_query = ""
    if mood == 'Happy':
        search_query = "Happy Upbeat Pop"
    elif mood == 'Sad':
        search_query = "Sad Piano Acoustic" 
    elif mood == 'Angry':
        search_query = "Heavy Metal Rock"
    elif mood == 'Neutral' or mood == 'Fear':
        search_query = "Lofi Chill Beats"
    elif mood == 'Surprise':
        search_query = "High Energy EDM"
    else:
        search_query = "Top Hits"

    try:
        # 2. Direct Track Search
        results = sp.search(q=search_query, limit=10, type='track')
        
        if results and results['tracks']['items']:
            items = results['tracks']['items']
            
            # Pick a random song from the results
            random_track = random.choice(items)
            
            song_name = random_track['name']
            artist_name = random_track['artists'][0]['name']
            song_url = random_track['external_urls']['spotify']
            
            print(f"\n--- PLAYING SONG ---")
            print(f"Title:  {song_name}")
            print(f"Artist: {artist_name}")
            print("[INFO] Opening in Browser...")
            
            webbrowser.open(song_url)
        else:
            print("[INFO] No songs found. That's unusual.")

    except Exception as e:
        print(f"\n[ERROR] Search failed. Error details: {e}")

# 4) Main Video Loop
cap = cv2.VideoCapture(0)

print("\n=================================================")
print("  MOOD MUSIC PLAYER READY!")
print("  1. Look at the camera.")
print("  2. PRESS 'SPACEBAR' to capture your mood.")
print("  3. Press 'q' to quit.")
print("=================================================\n")

last_mood = "Neutral"

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64))
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        prediction = emotion_classifier.predict(roi, verbose=0)[0]
        label = emotion_labels[prediction.argmax()]
        last_mood = label 

        cv2.putText(frame, f"Mood: {label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, "Press SPACE to get songs!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('Mood Music Player', frame)

    key = cv2.waitKey(1) & 0xFF
    
    if key == 32: # SPACEBAR
        play_mood_music(last_mood)
        time.sleep(2) 
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()