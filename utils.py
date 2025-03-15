from PIL import Image
import pygame
from speechbrain.pretrained import SpeakerRecognition
import os
import pyaudio
import wave
import subprocess
import edge_tts
from pydub import AudioSegment
from pydub.playback import play
import threading
import time
import re

''' ----- Dealing with GIFS ----- '''
def load_gif_frames(gif_path):
    """
    Loads frames from a GIF file and converts them to pygame Surfaces.
    Args:
        gif_path (str): The file path to the GIF file.
    Returns:
        list: A list of pygame Surface objects representing the frames of the GIF.
    """

    pil_image = Image.open(gif_path)
    frames = []

    try:
        while True:
            frame = pil_image.convert("RGBA")
            mode = frame.mode
            size = frame.size
            data = frame.tobytes()

            pygame_image = pygame.image.fromstring(data, size, mode) 
            frames.append(pygame.transform.scale(pygame_image, (720, 960)))
            
            pil_image.seek(pil_image.tell() + 1)  
    except EOFError:
        pass 

    return frames

def show_gif(gif_path):
    """
    Displays a GIF animation in a Pygame window.
    Args:
        gif_path (str): The file path to the GIF to be displayed.
    The function initializes Pygame, loads the frames of the GIF, and displays them in a loop
    until the window is closed by the user. The animation speed is controlled by a clock and
    a delay between frames.
    """
    
    pygame.init()
    screen = pygame.display.set_mode((720, 960))
    pygame.display.set_caption("Falando...")

    frames = load_gif_frames(gif_path)  
    clock = pygame.time.Clock()
    frame_index = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))
        screen.blit(frames[frame_index], (0, 0)) 
        pygame.display.flip()
        
        frame_index = (frame_index + 1) % len(frames) 
        clock.tick(10)  
        pygame.time.wait(200)

    pygame.quit()

''' -------------------------------- '''

''' Speaker recognition '''
def identify_speaker(voice_current_speaker): 
    """
    Identifies the speaker by comparing the given voice sample with pre-recorded voices.
    Args:
        voice_current_speaker (str): Path to the voice sample of the current speaker.
    Returns:
        str: The name of the identified speaker or "unknown" if the speaker is not recognized.
    This function uses a pre-trained speaker recognition model to compare the given voice sample
    with all voice samples in the "voices/" directory. It returns the name of the most similar
    voice sample. If the highest similarity score is below 0.1, it returns "unknown".
    """

    verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
    
    highest_score = 0
    highest_score_name = ""

    for v in os.listdir("voices/"):
       score, _ =  verification.verify_files(voice_current_speaker, f"voices/{v}")
       print(f"Comparing {voice_current_speaker} with voices/{v} with score {score}")

       if score.item() > highest_score:
           highest_score = score
           highest_score_name = v

    if highest_score < 0.1:
        highest_score_name = "unknown"

    print(f"O usuário identificado foi {highest_score_name} com a pontuação de {highest_score}")

    name, _ = highest_score_name.split(".")

    if name == "unknown":
        return highest_score_name
    else :
        return name
    
''' -------------------------------- '''

''' ----- Speech to text ----- '''
def get_input(calibration = False):
    """
    Records audio from the microphone, saves it to a WAV file, optionally identifies the speaker,
    and transcribes the audio using the Whisper model.
    Parameters:
    calibration (bool): If True, the function will not attempt to identify the speaker. Default is False.
    Returns:
    tuple: A tuple containing the transcribed text (str) and the speaker name (str or None).
    """

    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    WAVE_OUTPUT_FILENAME = "output.wav"
    RECORD_SECONDS = 5
    speaker_name = None
    should_save = False

    audio = pyaudio.PyAudio()

    ''' Recording '''
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("Gravando...")
    frames = []

    for _ in range (0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Gravação finalizada")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    ''' Saving '''
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    model_name = "tiny"
    subprocess.run([
        "whisper",
        "--language", "pt",
        "--word_timestamps", "True",
        "--model", model_name,
        "--output_dir", f"output-{model_name}",
        WAVE_OUTPUT_FILENAME
    ])

    with open(f"output-{model_name}/output.txt", "r") as f:
        text = f.read()
    
    if 'nome é' in text:
        should_save = True
    

    if not calibration:
        speaker_name = identify_speaker(WAVE_OUTPUT_FILENAME)

    if should_save:
        ''' If should_ve copy the audio in a folder called voices with the name of the user '''
        match = re.search(r"nome é (\w+)", text)
        name = match.group(1)
        os.makedirs(f"voices/", exist_ok=True)
        os.rename(WAVE_OUTPUT_FILENAME, f"voices/{name}.wav")    

    print(text)
    return text, speaker_name

''' -------------------------------- '''

''' ----- Text to speech ----- '''
def speak(text, gif_path):
    """
    Converts text to speech, plays the audio, and displays a GIF.
    Args:
        text (str): The text to be converted to speech.
        gif_path (str): The file path to the GIF to be displayed.
    Returns:
        None
    """

    TEXT = text
    VOICE = "pt-BR-ThalitaMultilingualNeural"
    OUTPUT = "output.mp3"

    comunicate = edge_tts.Communicate(TEXT, VOICE)
    
    with open(OUTPUT, "wb") as f:
        for chunk in comunicate.stream_sync():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
    
    sound = AudioSegment.from_mp3(OUTPUT)
    
    play_thread = threading.Thread(target=play, args=(sound,))
    play_thread.start()

    time.sleep(0.81)

    gif_thread = threading.Thread(target=show_gif, args=(gif_path,))
    gif_thread.start()

    play_thread.join()  
    pygame.event.post(pygame.event.Event(pygame.QUIT))  
    gif_thread.join()  