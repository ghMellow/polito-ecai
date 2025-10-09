import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np
import argparse
import os
from datetime import datetime
from pynput import keyboard
import queue
import threading

class AudioRecorder:
    def __init__(self, bit_depth, sampling_rate, duration):
        """Initialize the audio recorder with specified parameters"""
        self.bit_depth = np.int16 if bit_depth == 'int16' else np.int32
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.samples_per_file = sampling_rate * duration
        
        # Control flags
        self.recording = True
        self.storage_enabled = True
        
        # Audio buffer
        self.audio_queue = queue.Queue()
        self.current_buffer = []
        self.sample_count = 0
        
        print(f"\n{'='*60}")
        print(f"Audio Recorder Initialized")
        print(f"{'='*60}")
        print(f"Bit Depth: {bit_depth}")
        print(f"Sampling Rate: {sampling_rate} Hz")
        print(f"Recording Duration per file: {duration} second(s)")
        print(f"Samples per file: {self.samples_per_file}")
        print(f"\nControls:")
        print(f"  Q - Stop recording and exit")
        print(f"  P - Toggle storage (enable/disable saving)")
        print(f"\nStorage: {'ENABLED' if self.storage_enabled else 'DISABLED'}")
        print(f"{'='*60}\n")
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback function called by sounddevice for each audio block"""
        if status:
            print(f"Status: {status}")
        
        # Add audio data to buffer
        self.current_buffer.append(indata.copy())
        self.sample_count += len(indata)
        
        # Check if we have enough samples for one file
        if self.sample_count >= self.samples_per_file:
            # Combine buffer into single array
            audio_data = np.concatenate(self.current_buffer[:], axis=0)
            
            # Trim to exact duration
            audio_data = audio_data[:self.samples_per_file]
            
            # Add to queue for saving
            if self.storage_enabled:
                self.audio_queue.put(audio_data)
            
            # Reset buffer, keeping any extra samples
            remaining_samples = self.sample_count - self.samples_per_file
            if remaining_samples > 0:
                self.current_buffer = [self.current_buffer[-1][-remaining_samples:]]
                self.sample_count = remaining_samples
            else:
                self.current_buffer = []
                self.sample_count = 0
    
    def save_audio_worker(self):
        """Worker thread to save audio files"""
        while self.recording or not self.audio_queue.empty():
            try:
                # Get audio data from queue with timeout
                audio_data = self.audio_queue.get(timeout=0.5)
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"audio_{timestamp}.wav"
                
                # Save audio file
                wavfile.write(filename, self.sampling_rate, audio_data.astype(self.bit_depth))
                
                # Get file size
                file_size_bytes = os.path.getsize(filename)
                file_size_kb = file_size_bytes / 1024
                
                print(f"[SAVED] {filename} - Size: {file_size_kb:.2f} KB ({file_size_bytes} bytes)")
                
                self.audio_queue.task_done()
            except queue.Empty:
                continue
    
    def on_press(self, key):
        """Handle keyboard input"""
        try:
            if hasattr(key, 'char'):
                if key.char == 'q' or key.char == 'Q':
                    print("\n[Q pressed] Stopping recording...")
                    self.recording = False
                    return False  # Stop listener
                elif key.char == 'p' or key.char == 'P':
                    self.storage_enabled = not self.storage_enabled
                    status = "ENABLED" if self.storage_enabled else "DISABLED"
                    print(f"\n[P pressed] Storage {status}")
        except AttributeError:
            pass
    
    def start_recording(self):
        """Start the audio recording process"""
        # Start the file saving thread
        save_thread = threading.Thread(target=self.save_audio_worker, daemon=True)
        save_thread.start()
        
        # Start keyboard listener
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()
        
        try:
            # Start audio stream
            with sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sampling_rate,
                dtype=self.bit_depth
            ):
                print("Recording started... Press Q to stop, P to toggle storage\n")
                
                # Keep recording until stopped
                while self.recording:
                    sd.sleep(100)
                
        except KeyboardInterrupt:
            print("\n\nRecording interrupted by user")
        finally:
            print("\nStopping recording...")
            self.recording = False
            
            # Wait for all files to be saved
            print("Waiting for remaining files to be saved...")
            self.audio_queue.join()
            save_thread.join(timeout=2)
            
            print("Recording stopped successfully")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Record audio with USB microphone on Raspberry Pi',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 lab1_ex2.py --bit_depth int16 --sampling_rate 44100 --duration 1
  python3 lab1_ex2.py --bit_depth int32 --sampling_rate 48000 --duration 2
        """
    )
    
    parser.add_argument(
        '--bit_depth',
        type=str,
        choices=['int16', 'int32'],
        required=True,
        help='Bit depth for audio recording (int16 or int32)'
    )
    
    parser.add_argument(
        '--sampling_rate',
        type=int,
        required=True,
        help='Sampling rate in Hertz (e.g., 44100, 48000)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        required=True,
        help='Recording duration per file in seconds'
    )
    
    args = parser.parse_args()
    
    # Validate sampling rate
    if args.sampling_rate <= 0:
        parser.error("Sampling rate must be positive")
    
    if args.duration <= 0:
        parser.error("Duration must be positive")
    
    # Create and start recorder
    recorder = AudioRecorder(args.bit_depth, args.sampling_rate, args.duration)
    recorder.start_recording()


if __name__ == "__main__":
    main()
