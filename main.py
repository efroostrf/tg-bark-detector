import os
import sys
import time
import numpy as np
import asyncio
import threading
import tempfile
import io
import wave
import subprocess
import csv
import base64
import datetime
import json
from collections import deque
from telegram import Bot
from telegram.error import TelegramError
from dotenv import load_dotenv
import pymongo
from pymongo import MongoClient
from bson.binary import Binary
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
from gi.repository import Gst, GLib

# Load environment variables
load_dotenv()

# Get RTSP URL from environment variable
RTSP_URL = os.getenv('RTSP_URL')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_USER_IDS = os.getenv('TELEGRAM_USER_IDS')
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'bark_system')

if not RTSP_URL:
    print("Error: RTSP_URL environment variable is not set")
    sys.exit(1)

if not TELEGRAM_BOT_TOKEN:
    print("Warning: TELEGRAM_BOT_TOKEN environment variable is not set. Notifications will be disabled.")

if not TELEGRAM_USER_IDS:
    print("Warning: TELEGRAM_USER_IDS environment variable is not set. Notifications will be disabled.")

# Lazy-load TensorFlow to avoid startup delay
tf = None
hub = None
yamnet_model = None
yamnet_classes = None

# Initialize MongoDB client (lazily)
mongodb_client = None
db = None

def get_mongodb():
    """Initialize MongoDB connection lazily"""
    global mongodb_client, db
    
    if mongodb_client is None:
        try:
            print(f"Connecting to MongoDB at {MONGODB_URI}...")
            mongodb_client = MongoClient(MONGODB_URI)
            db = mongodb_client[MONGODB_DB_NAME]
            
            # Create collections if they don't exist
            if 'noise_detections' not in db.list_collection_names():
                db.create_collection('noise_detections')
                print("Created 'noise_detections' collection")
            
            # Create indexes for faster queries
            db.noise_detections.create_index([('timestamp', pymongo.DESCENDING)])
            db.noise_detections.create_index([('is_dog_bark', pymongo.ASCENDING)])
            
            print(f"Connected to MongoDB database: {MONGODB_DB_NAME}")
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            mongodb_client = None
            db = None
    
    return db

def load_tf_model():
    """Load TensorFlow and YAMNet model lazily"""
    global tf, hub, yamnet_model, yamnet_classes
    
    if yamnet_model is not None:
        return yamnet_model, yamnet_classes
    
    print("Loading TensorFlow and YAMNet model...")
    
    try:
        # First import tensorflow
        import tensorflow as tf
        import tensorflow_hub as hub

        # Disable GPU if there are issues
        try:
            # Check for GPUs
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"Found {len(gpus)} GPU(s): {gpus}")
                
                # Try to initialize GPU, but catch any errors
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print("GPU memory growth enabled")
                except RuntimeError as e:
                    print(f"Error configuring GPU: {e}")
                    print("Disabling GPU usage...")
                    # Disable GPU usage completely
                    tf.config.set_visible_devices([], 'GPU')
        except:
            print("No GPUs found or error listing GPUs, falling back to CPU")
            # Force CPU mode
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        # Log which device is being used
        print("TensorFlow will run on:", tf.config.list_physical_devices())
        
        # Load the YAMNet model
        print("Loading YAMNet model from TensorFlow Hub (this may take a moment)...")
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        
        # Load class names
        yamnet_class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
        with open(yamnet_class_map_path) as csv_file:
            reader = csv.DictReader(csv_file)
            yamnet_classes = {int(row['index']): row['display_name'] for row in reader}
        
        print(f"YAMNet model loaded successfully with {len(yamnet_classes)} sound classes")
        
        return yamnet_model, yamnet_classes
    except Exception as e:
        print(f"Error loading TensorFlow or YAMNet model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Configuration for dog bark detection
class Config:
    SAMPLE_RATE = 16000  # Sample rate we're converting to
    ENERGY_THRESHOLD = 0.01  # Adjust based on your testing
    BARK_FREQUENCY_MULTIPLIER = 10  # Adjust based on your testing
    DETECTION_COOLDOWN = 7.0  # Seconds between detections to avoid duplicates
    
    # Audio buffer configuration (for voice messages)
    AUDIO_BUFFER_SIZE_SECONDS = 7.0  # Total seconds of audio to keep in buffer
    AUDIO_CHUNK_SIZE = 4096  # Exact size of audio chunks in bytes
    
    # AI model configuration
    DOG_SOUND_CLASSES = [
        "Dog", "Bark", "Howl", "Bow-wow", "Growling", "Whimper (dog)", 
        "Bay", "Yelp, dog", "Canidae, dogs, wolves", "Domestic animals, pets", "Animal", "Wild animals"
    ]
    MIN_DOG_SCORE = 0.25  # Minimum confidence score to confirm dog bark
    MAX_RESULTS = 10  # Top N results to consider
    
    # MongoDB configuration
    STORE_ALL_NOISE_DETECTIONS = True  # Whether to store all noise detections or only dog barks
    STORE_AUDIO_IN_DB = True  # Whether to store audio data in DB (can make DB large)
    MAX_STORED_DETECTIONS = 1000  # Maximum number of detections to keep in DB

class MongoDBLogger:
    def __init__(self):
        self.db = None
    
    def ensure_db_connection(self):
        if self.db is None:
            self.db = get_mongodb()
        return self.db is not None
    
    def log_noise_detection(self, audio_data, energy, frequency_energy, predictions=None, is_dog_bark=False, dog_classes=None):
        """Log a noise detection to MongoDB"""
        if not self.ensure_db_connection():
            print("Cannot log noise detection: no MongoDB connection")
            return
        
        # Only store if configured to store all detections or if it's a dog bark
        if not Config.STORE_ALL_NOISE_DETECTIONS and not is_dog_bark:
            return
        
        try:
            # Create document
            timestamp = datetime.datetime.now()
            document = {
                "timestamp": timestamp,
                "timestamp_str": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "energy": float(energy),
                "frequency_energy": float(frequency_energy),
                "is_dog_bark": is_dog_bark,
                "predictions": predictions if predictions else [],
            }
            
            # Add dog classes if this is a dog bark
            if is_dog_bark and dog_classes:
                document["dog_classes"] = dog_classes
            
            # Store audio data if configured
            if Config.STORE_AUDIO_IN_DB and audio_data is not None:
                # Store as Binary data
                document["audio_data"] = Binary(audio_data)
                document["audio_format"] = "wav"
                document["sample_rate"] = Config.SAMPLE_RATE
            
            # Insert into database
            result = self.db.noise_detections.insert_one(document)
            
            # Manage collection size
            self.enforce_collection_limit()
            
            print(f"Logged noise detection to MongoDB (ID: {result.inserted_id})")
            return result.inserted_id
        except Exception as e:
            print(f"Error logging to MongoDB: {e}")
            return None
    
    def enforce_collection_limit(self):
        """Make sure we don't store too many documents"""
        try:
            # Count documents
            count = self.db.noise_detections.count_documents({})
            
            # If we have too many, delete the oldest ones
            if count > Config.MAX_STORED_DETECTIONS:
                # How many to delete
                to_delete = count - Config.MAX_STORED_DETECTIONS
                
                # Get the oldest documents
                oldest = list(self.db.noise_detections.find().sort("timestamp", pymongo.ASCENDING).limit(to_delete))
                
                if oldest:
                    # Delete them
                    ids_to_delete = [doc["_id"] for doc in oldest]
                    result = self.db.noise_detections.delete_many({"_id": {"$in": ids_to_delete}})
                    print(f"Deleted {result.deleted_count} old noise detections from MongoDB")
        except Exception as e:
            print(f"Error enforcing collection limit: {e}")

class TelegramNotifier:
    def __init__(self, token, user_ids_str):
        self.token = token
        self.user_ids = []

        # Parse user IDs from comma-separated string
        if user_ids_str:
            try:
                self.user_ids = [int(user_id.strip()) for user_id in user_ids_str.split(',')]
                print(f"Initialized Telegram notifier for users: {self.user_ids}")
            except ValueError:
                print("Error: TELEGRAM_USER_IDS should be comma-separated integers")
        
        self.bot = Bot(token=token) if token else None
        self.loop = asyncio.new_event_loop()
        self.thread = None
    
    def start(self):
        if not self.bot or not self.user_ids:
            print("Telegram notifications disabled: missing token or user IDs")
            return False
        
        def run_async_loop():
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        
        self.thread = threading.Thread(target=run_async_loop, daemon=True)
        self.thread.start()
        return True
    
    def send_notification(self, message):
        if not self.bot or not self.user_ids:
            return
        
        async def send_messages():
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            full_message = f"🔔 {timestamp}\n{message}"
            
            for user_id in self.user_ids:
                try:
                    await self.bot.send_message(chat_id=user_id, text=full_message)
                    print(f"Notification sent to user {user_id}")
                except TelegramError as e:
                    print(f"Failed to send notification to user {user_id}: {e}")
        
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(send_messages(), self.loop)
    
    def send_voice_message(self, audio_file_path, caption):
        if not self.bot or not self.user_ids:
            return
        
        async def send_audio():
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            full_caption = f"🔔 {timestamp}\n{caption}"
            
            for user_id in self.user_ids:
                try:
                    with open(audio_file_path, 'rb') as audio_file:
                        message = await self.bot.send_voice(
                            chat_id=user_id, 
                            voice=audio_file, 
                            caption=full_caption,
                            # Add duration to help Telegram
                            duration=int(os.path.getsize(audio_file_path) / 8000)  # rough estimate
                        )
                    print(f"Voice message sent to user {user_id}")
                except TelegramError as e:
                    print(f"Failed to send voice message to user {user_id}: {e}")
        
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(send_audio(), self.loop)

class AudioBuffer:
    def __init__(self):
        self.sample_rate = Config.SAMPLE_RATE
        self.channels = 1
        self.sample_width = 2  # 16-bit audio (2 bytes)
        
        # Calculate buffer size in chunks
        # How many 16-bit samples in a chunk
        self.samples_per_chunk = Config.AUDIO_CHUNK_SIZE // 2  # 16-bit = 2 bytes per sample
        
        # How many chunks to store in buffer
        # total_samples = sample_rate * seconds
        # total_chunks = total_samples / samples_per_chunk
        self.buffer_chunks = int(Config.AUDIO_BUFFER_SIZE_SECONDS * 
                               self.sample_rate / self.samples_per_chunk)
        
        # Create circular buffer for audio
        self.audio_buffer = deque(maxlen=self.buffer_chunks)
        
        # State tracking
        self.bark_detected = False
        self.chunks_after_bark = 0
        self.chunks_needed_after_bark = self.buffer_chunks // 2  # Half the buffer size
    
    def add_chunk(self, audio_data):
        """Add audio chunk to the buffer"""
        # Always add to circular buffer
        self.audio_buffer.append(audio_data)
        
        # If we're in post-bark collection mode, count chunks
        if self.bark_detected:
            self.chunks_after_bark += 1
            
            # Check if we've collected enough chunks after bark
            if self.chunks_after_bark >= self.chunks_needed_after_bark:
                self.bark_detected = False
                self.chunks_after_bark = 0
                return True  # Buffer ready to send
        
        return False  # Still collecting
    
    def mark_bark_detected(self):
        """Mark that a bark was detected, reset post-bark counter"""
        self.bark_detected = True
        self.chunks_after_bark = 0
    
    def save_to_wav(self, filename):
        """Save the full audio buffer to a WAV file"""
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(self.sample_width)
            wav_file.setframerate(self.sample_rate)
            
            # Write all buffer chunks
            for chunk in self.audio_buffer:
                wav_file.writeframes(chunk)
        
        print(f"Saved audio to {filename} ({len(self.audio_buffer)} chunks)")
        return filename
    
    def get_wav_bytes(self):
        """Get the audio buffer as WAV file bytes"""
        with io.BytesIO() as wav_io:
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.sample_width)
                wav_file.setframerate(self.sample_rate)
                
                # Write all buffer chunks
                for chunk in self.audio_buffer:
                    wav_file.writeframes(chunk)
            
            return wav_io.getvalue()
    
    def save_to_ogg(self, filename_base):
        """Save the full audio buffer to an OGG file (for Telegram)"""
        # First save as WAV
        wav_filename = f"{filename_base}.wav"
        self.save_to_wav(wav_filename)
        
        # Then convert to OGG using ffmpeg (better for Telegram)
        ogg_filename = f"{filename_base}.ogg"
        try:
            subprocess.run([
                'ffmpeg', '-y',
                '-i', wav_filename,
                '-c:a', 'libopus',
                '-b:a', '64k',
                ogg_filename
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Delete the intermediate WAV file
            os.remove(wav_filename)
            
            print(f"Converted to OGG: {ogg_filename}")
            return ogg_filename
        except subprocess.CalledProcessError as e:
            print(f"Failed to convert to OGG: {e}")
            print(f"FFMPEG stderr: {e.stderr.decode() if e.stderr else 'None'}")
            
            # Fall back to WAV if conversion fails
            return wav_filename
        except FileNotFoundError:
            print("FFMPEG not found, using WAV format instead")
            return wav_filename

    def get_duration(self):
        """Get the approximate duration of the buffer in seconds"""
        return len(self.audio_buffer) * self.samples_per_chunk / self.sample_rate
    
    def get_buffer_info(self):
        """Get information about the buffer for debugging"""
        return {
            "total_chunks": len(self.audio_buffer),
            "max_chunks": self.buffer_chunks,
            "chunk_duration": self.samples_per_chunk / self.sample_rate,
            "total_duration": self.get_duration(),
            "bytes_per_chunk": Config.AUDIO_CHUNK_SIZE
        }
    
    def get_audio_samples(self):
        """Get the audio samples as a single numpy array for model inference"""
        # Create a single array to hold all audio samples
        total_samples = len(self.audio_buffer) * self.samples_per_chunk
        audio_samples = np.zeros(total_samples, dtype=np.float32)
        
        # Fill the array with audio samples from the buffer
        for i, chunk in enumerate(self.audio_buffer):
            # Convert from bytes to int16
            int_samples = np.frombuffer(chunk, dtype=np.int16)
            # Convert to float32 in range [-1.0, 1.0]
            float_samples = int_samples.astype(np.float32) / 32768.0
            # Copy to the appropriate position in the output array
            start_idx = i * self.samples_per_chunk
            end_idx = start_idx + len(float_samples)
            audio_samples[start_idx:end_idx] = float_samples
        
        return audio_samples

class AudioAnalyzer:
    def __init__(self, telegram_notifier=None):
        self.last_detection_time = 0
        self.telegram_notifier = telegram_notifier
        self.audio_buffer = AudioBuffer()
        self.noise_detected = False
        self.pending_analysis = False
        self.mongodb_logger = MongoDBLogger()
    
    def analyze_audio(self, audio_data):
        """Analyze audio data for potential noise (first stage detection)"""
        # Add audio chunk to buffer regardless of detection
        buffer_ready = self.audio_buffer.add_chunk(audio_data)
        
        # If enough post-detection data has been collected, analyze the full buffer
        if buffer_ready and self.noise_detected:
            self.pending_analysis = False
            self.noise_detected = False
            self.analyze_buffer_for_dog_bark()
            return False
        
        # If we're already in the process of collecting post-detection audio or pending analysis
        if self.noise_detected or self.pending_analysis:
            return False
        
        # Convert audio data from bytes to int16 samples
        int_data = np.frombuffer(audio_data, dtype=np.int16)
        
        # Normalize to -1.0 to 1.0 float range
        float_data = int_data.astype(np.float32) / 32768.0
        
        # Simple energy-based detection
        energy = np.mean(np.abs(float_data))
        
        # Print debug info for energy levels
        print(f"Current audio energy: {energy:.6f} (threshold: {Config.ENERGY_THRESHOLD})")
        
        # Check if we're in cooldown period
        current_time = time.time()
        if current_time - self.last_detection_time < Config.DETECTION_COOLDOWN:
            return False
        
        if energy > Config.ENERGY_THRESHOLD:
            # Calculate frequency components
            fft_data = np.abs(np.fft.fft(float_data))
            freqs = np.fft.fftfreq(len(float_data), 1/Config.SAMPLE_RATE)
            
            # Dog barks typically have strong components between 500-4000 Hz
            bark_range = (freqs >= 500) & (freqs <= 4000)
            bark_energy = np.mean(fft_data[bark_range])
            bark_threshold = Config.ENERGY_THRESHOLD * Config.BARK_FREQUENCY_MULTIPLIER
            
            # Print debug info for frequency analysis
            print(f"Bark frequency energy: {bark_energy:.6f} (threshold: {bark_threshold})")
            
            if bark_energy > bark_threshold:
                self.last_detection_time = current_time
                self.on_noise_detected(energy, bark_energy)
                return True
        
        return False
    
    def on_noise_detected(self, energy, bark_energy):
        """Called when initial noise detection happens"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"🔊 NOISE DETECTED at {timestamp}! Starting AI analysis...")
        print(f"Audio energy: {energy:.6f}, Frequency energy: {bark_energy:.6f}")
        
        # Start the post-detection collection process
        self.audio_buffer.mark_bark_detected()
        self.noise_detected = True
        self.pending_analysis = True
        
        # Get buffer info for debugging
        buffer_info = self.audio_buffer.get_buffer_info()
        print(f"Audio buffer info: {buffer_info}")
    
    def analyze_buffer_for_dog_bark(self):
        """Use YAMNet to analyze the audio buffer for dog barks"""
        print("Running AI analysis on the audio buffer...")
        
        # Get audio samples from buffer
        waveform = self.audio_buffer.get_audio_samples()
        
        # Load model if needed
        model, class_names = load_tf_model()
        if model is None:
            print("Cannot analyze audio: model not loaded")
            return
        
        try:
            # Run inference with YAMNet
            scores, embeddings, spectrogram = model(waveform)
            scores = scores.numpy()
            
            # Get top predictions
            top_indices = np.argsort(np.mean(scores, axis=0))[-Config.MAX_RESULTS:][::-1]
            top_scores = np.mean(scores, axis=0)[top_indices]
            
            # Format predictions for logging and display
            predictions = []
            for i, (index, score) in enumerate(zip(top_indices, top_scores)):
                class_name = class_names[index]
                predictions.append({
                    "rank": i+1,
                    "class_name": class_name,
                    "score": float(score),
                    "index": int(index)
                })
                
            # Check for dog sounds
            is_dog_bark = False
            dog_classes_found = []
            
            print("\nTop sound detections:")
            for pred in predictions:
                print(f"{pred['rank']}. {pred['class_name']}: {pred['score']:.3f}")
                
                # Check if this is a dog-related sound
                for dog_class in Config.DOG_SOUND_CLASSES:
                    if dog_class.lower() in pred['class_name'].lower() and pred['score'] >= Config.MIN_DOG_SCORE:
                        is_dog_bark = True
                        dog_classes_found.append(f"{pred['class_name']} ({pred['score']:.3f})")
            
            # Get WAV bytes for logging
            audio_wav = self.audio_buffer.get_wav_bytes() if Config.STORE_AUDIO_IN_DB else None
            
            # Log detection to MongoDB
            energy = float(np.mean(np.abs(waveform)))
            frequency_energy = float(np.mean(np.abs(np.fft.fft(waveform))[500:4000]))
            self.mongodb_logger.log_noise_detection(
                audio_data=audio_wav,
                energy=energy,
                frequency_energy=frequency_energy,
                predictions=predictions,
                is_dog_bark=is_dog_bark,
                dog_classes=dog_classes_found if is_dog_bark else None
            )
            
            if is_dog_bark:
                print(f"🐶 DOG BARK CONFIRMED BY AI! Classes: {', '.join(dog_classes_found)}")
                self.on_bark_detected(dog_classes_found)
            else:
                print("❌ No dog bark detected by AI, discarding...")
                
        except Exception as e:
            print(f"Error during AI analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def on_bark_detected(self, dog_classes):
        """Called when a bark is confirmed by the AI model"""
        # Create temporary file base name (without extension)
        temp_dir = tempfile.gettempdir()
        base_filename = os.path.join(temp_dir, f"bark_audio_{int(time.time())}")
        
        # Save the audio buffer to OGG file (preferred for Telegram)
        try:
            audio_path = self.audio_buffer.save_to_ogg(base_filename)
        except Exception as e:
            print(f"Error saving OGG file: {e}")
            # Fallback to WAV if OGG fails
            audio_path = self.audio_buffer.save_to_wav(f"{base_filename}.wav")
        
        # Send as voice message
        if self.telegram_notifier:
            duration = self.audio_buffer.get_duration()
            caption = f"🐶 DOG BARK CONFIRMED!\n\nDuration: {duration:.1f}s\nAI detected: {', '.join(dog_classes)}"
            self.telegram_notifier.send_voice_message(audio_path, caption)
            
            # Delete the temporary file after a delay (to ensure it's sent)
            def delete_after_delay(path, delay=10):
                time.sleep(delay)
                try:
                    os.remove(path)
                    print(f"Deleted temporary file: {path}")
                except Exception as e:
                    print(f"Failed to delete temporary file {path}: {e}")
            
            threading.Thread(target=delete_after_delay, args=(audio_path,), daemon=True).start()

# Global pipeline variable for use in callbacks
pipeline = None
audio_detected = False

def on_message(bus, message, loop):
    msg_type = message.type
    if msg_type == Gst.MessageType.EOS:
        print("End of stream")
        loop.quit()
    elif msg_type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err.message}")
        print(f"Debug info: {debug}")
        loop.quit()
    elif msg_type == Gst.MessageType.WARNING:
        warn, debug = message.parse_warning()
        print(f"Warning: {warn.message}")
        print(f"Debug info: {debug}")
    elif msg_type == Gst.MessageType.STATE_CHANGED:
        if message.src == pipeline:
            old_state, new_state, pending_state = message.parse_state_changed()
            print(f"Pipeline state changed from {old_state.value_nick} to {new_state.value_nick}")

def on_new_sample(sink):
    try:
        sample = sink.emit("pull-sample")
        if sample:
            buf = sample.get_buffer()
            result, map_info = buf.map(Gst.MapFlags.READ)
            if result:
                audio_data = map_info.data  # raw PCM bytes
                global audio_detected
                if not audio_detected:
                    audio_detected = True
                    print(f"🎉 AUDIO DETECTED! Got first audio chunk of {len(audio_data)} bytes")
                else:
                    print(f"Processing audio chunk of {len(audio_data)} bytes")
                
                # Process audio data for bark detection
                analyzer.analyze_audio(audio_data)
                
                buf.unmap(map_info)
            return Gst.FlowReturn.OK
        return Gst.FlowReturn.ERROR
    except Exception as e:
        print(f"Error in on_new_sample: {str(e)}")
        return Gst.FlowReturn.ERROR

def main():
    # Initialize GStreamer
    Gst.init(None)
    
    # Check for ffmpeg availability
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("FFMPEG found, will use OGG format for voice messages")
    except FileNotFoundError:
        print("FFMPEG not found, will use WAV format for voice messages (may not show duration in Telegram)")
        print("To fix: install FFMPEG using 'sudo apt-get install ffmpeg' or equivalent")
    
    # Initialize MongoDB in background to not delay startup
    def init_mongodb_background():
        try:
            db = get_mongodb()
            if db is not None:
                print("MongoDB initialized successfully")
            else:
                print("Warning: Failed to initialize MongoDB. Logging will be disabled.")
        except Exception as e:
            print(f"Error initializing MongoDB: {e}")
    
    threading.Thread(target=init_mongodb_background, daemon=True).start()
    
    # Initialize Telegram notifier
    telegram_notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_USER_IDS)
    if telegram_notifier.start():
        print("Telegram notifications enabled")
    
    # Start loading the model in a background thread
    def load_model_background():
        print("Loading TensorFlow and YAMNet in background thread...")
        try:
            load_tf_model()
            print("Background model loading complete")
        except Exception as e:
            print(f"Error in background model loading: {e}")
    
    threading.Thread(target=load_model_background, daemon=True).start()
    
    # Initialize analyzer with Telegram notifier
    global analyzer
    analyzer = AudioAnalyzer(telegram_notifier)
    
    # Test notification on startup
    if TELEGRAM_BOT_TOKEN and TELEGRAM_USER_IDS:
        telegram_notifier.send_notification("🎧 Bark detector started with AI verification and MongoDB logging")
    
    try:
        # Create GStreamer pipeline using parse_launch for simplicity and reliability
        global pipeline

        # Create a pipeline specifically for A-Law audio (8kHz)
        pipeline_str = f'''
        rtspsrc location={RTSP_URL} latency=0 buffer-mode=none ! 
        application/x-rtp,media=audio,encoding-name=PCMA,clock-rate=8000 ! 
        rtppcmadepay ! alawdec ! 
        audioconvert ! audioresample ! 
        audio/x-raw,format=S16LE,channels=1,rate=16000 ! 
        appsink name=audio_sink emit-signals=true sync=false max-buffers=10 drop=true
        '''
        
        print(f"Trying to connect to RTSP stream with A-Law audio: {RTSP_URL}")
        print("Using pipeline:\n", pipeline_str)
        
        pipeline = Gst.parse_launch(pipeline_str)
        audio_sink = pipeline.get_by_name("audio_sink")
        audio_sink.connect("new-sample", on_new_sample)
        
        # Set up bus to handle messages
        bus = pipeline.get_bus()
        loop = GLib.MainLoop()
        bus.add_signal_watch()
        bus.connect("message", on_message, loop)
        
        # Start playing
        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Failed to start A-Law pipeline, trying alternate pipeline...")
            pipeline.set_state(Gst.State.NULL)
            
            # Try alternative pipeline with selection
            pipeline_str = f'''
            rtspsrc location={RTSP_URL} latency=0 ! 
            application/x-rtp,media=audio,payload=8 ! 
            rtpjitterbuffer ! rtppcmadepay ! alawdec ! 
            audioconvert ! audioresample ! 
            audio/x-raw,format=S16LE,channels=1,rate=16000 ! 
            appsink name=audio_sink emit-signals=true sync=false max-buffers=10 drop=true
            '''
            
            print("Using alternate A-Law pipeline:\n", pipeline_str)
            pipeline = Gst.parse_launch(pipeline_str)
            audio_sink = pipeline.get_by_name("audio_sink")
            audio_sink.connect("new-sample", on_new_sample)
            
            # Set up bus again
            bus = pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", on_message, loop)
            
            ret = pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                print("All pipeline attempts failed. Your camera may not support audio streaming.")
                return
        
        print("Pipeline started, waiting for audio...")
        
        # Wait for some time to see if we get audio
        timeout = 15  # seconds
        start_time = time.time()
        
        try:
            while time.time() - start_time < timeout and not audio_detected:
                loop.get_context().iteration(False)  # Non-blocking iteration
                time.sleep(0.1)
                
            if not audio_detected:
                print(f"No audio detected after {timeout} seconds.")
                print("Final attempt with manual pad connection:")
                
                pipeline.set_state(Gst.State.NULL)
                
                # Use a different approach with manual pad connection
                pipeline = Gst.Pipeline.new("pipeline")
                
                # Create elements
                src = Gst.ElementFactory.make("rtspsrc", "src")
                src.set_property("location", RTSP_URL)
                src.set_property("latency", 0)
                src.set_property("buffer-mode", 0)  # None
                
                # Create audio processing elements
                depay = Gst.ElementFactory.make("rtppcmadepay", "depay")
                decode = Gst.ElementFactory.make("alawdec", "decode")
                convert = Gst.ElementFactory.make("audioconvert", "convert")
                resample = Gst.ElementFactory.make("audioresample", "resample")
                capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
                caps = Gst.Caps.from_string("audio/x-raw,format=S16LE,channels=1,rate=16000")
                capsfilter.set_property("caps", caps)
                sink = Gst.ElementFactory.make("appsink", "sink")
                sink.set_property("emit-signals", True)
                sink.set_property("sync", False)
                sink.set_property("max-buffers", 10)
                sink.set_property("drop", True)
                sink.connect("new-sample", on_new_sample)
                
                # Add all elements to pipeline
                for element in [src, depay, decode, convert, resample, capsfilter, sink]:
                    pipeline.add(element)
                
                # Link static elements (except for rtspsrc)
                if not Gst.Element.link_many(depay, decode, convert, resample, capsfilter, sink):
                    print("Failed to link elements")
                    return
                
                # Connect to pad-added signal
                def on_pad_added(src, pad):
                    sink_pad = depay.get_static_pad("sink")
                    if not sink_pad.is_linked():
                        pad.link(sink_pad)
                        print("Linked RTSP audio pad manually")
                    
                src.connect("pad-added", on_pad_added)
                
                # Set up bus
                bus = pipeline.get_bus()
                bus.add_signal_watch()
                bus.connect("message", on_message, loop)
                
                # Start playing
                print("Starting final pipeline attempt...")
                ret = pipeline.set_state(Gst.State.PLAYING)
                if ret == Gst.StateChangeReturn.FAILURE:
                    print("Failed to start final pipeline attempt")
                    return
                
                # Wait again for audio
                timeout = 15
                start_time = time.time()
                while time.time() - start_time < timeout and not audio_detected:
                    loop.get_context().iteration(False)
                    time.sleep(0.1)
                
                if not audio_detected:
                    print("Still no audio detected. Please ensure your camera's RTSP stream supports audio streaming.")
                    return
                
            # Continue running if audio was detected
            print("Audio stream detected. Running the main loop...")
            print("Dog bark detection is now active. Listening for barks...")
            loop.run()
        except KeyboardInterrupt:
            print("Interrupted")
            if TELEGRAM_BOT_TOKEN and TELEGRAM_USER_IDS:
                telegram_notifier.send_notification("⛔️ Bark detector has been stopped")
        finally:
            pipeline.set_state(Gst.State.NULL)
    except Exception as e:
        print(f"Error setting up pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()