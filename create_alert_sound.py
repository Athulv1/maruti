"""
Generate a violation alert sound file with voice
"""
import os
import subprocess

def create_violation_alert():
    """Create a voice alert saying 'Violation Found'"""
    
    print("Generating voice alert: 'Violation Found'")
    
    # Use espeak to generate the voice
    temp_wav = "temp_alert.wav"
    
    try:
        # Generate voice with espeak
        # -s 150: speed (words per minute)
        # -a 200: amplitude (volume)
        # -v en: English voice
        subprocess.run([
            'espeak',
            '-s', '150',
            '-a', '200',
            '-v', 'en',
            '-w', temp_wav,
            'Violation Found'
        ], check=True)
        
        # Convert to proper format with ffmpeg
        subprocess.run([
            'ffmpeg',
            '-i', temp_wav,
            '-ar', '44100',
            '-ac', '1',
            '-y',
            'violation_alert.wav'
        ], check=True, capture_output=True)
        
        # Clean up temp file
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        
        print("✓ Created violation_alert.wav with voice: 'Violation Found'")
        
    except subprocess.CalledProcessError as e:
        print(f"Error creating voice alert: {e}")
        print("Falling back to beep sound...")
        create_beep_alert()
    except FileNotFoundError:
        print("espeak or ffmpeg not found. Installing...")
        print("Please run: sudo apt-get install espeak ffmpeg")
        create_beep_alert()

def create_beep_alert():
    """Fallback: Create a simple beep alert sound"""
    try:
        import numpy as np
        from scipy.io import wavfile
        
        sample_rate = 44100
        duration = 1.0
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Two-tone siren
        tone1 = np.sin(2 * np.pi * 1000 * t[:len(t)//2])
        tone2 = np.sin(2 * np.pi * 1500 * t[len(t)//2:])
        
        alert = np.concatenate([tone1, tone2])
        
        # Fade in/out
        fade_samples = int(sample_rate * 0.05)
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        alert[:fade_samples] *= fade_in
        alert[-fade_samples:] *= fade_out
        
        # Normalize
        alert = alert / np.max(np.abs(alert))
        alert = (alert * 32767).astype(np.int16)
        
        wavfile.write('violation_alert.wav', sample_rate, alert)
        print("✓ Created violation_alert.wav (beep sound)")
    except Exception as e:
        print(f"Error creating beep alert: {e}")

if __name__ == "__main__":
    create_violation_alert()
