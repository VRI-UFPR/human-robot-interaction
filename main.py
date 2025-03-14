from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import db_handler
from tool_reminder import ToolReminder
from langchain.agents import AgentType, initialize_agent
from langchain.agents import Tool
import datetime
import edge_tts
from pydub import AudioSegment
from pydub.playback import play
import os
import pyaudio
import wave
import subprocess
import pygame
import threading
from PIL import Image
import time
from speechbrain.pretrained import SpeakerRecognition

def load_gif_frames(gif_path):
    """Carrega os quadros do GIF e os converte para Surface do pygame."""
    pil_image = Image.open(gif_path)
    frames = []

    try:
        while True:
            frame = pil_image.convert("RGBA")  # Converte para RGBA
            mode = frame.mode
            size = frame.size
            data = frame.tobytes()

            pygame_image = pygame.image.fromstring(data, size, mode)  # Converte para pygame Surface
            frames.append(pygame.transform.scale(pygame_image, (720, 960)))  # Redimensiona
            
            pil_image.seek(pil_image.tell() + 1)  # Avança para o próximo quadro
    except EOFError:
        pass  # Sai quando não há mais quadros

    return frames

def get_input():
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    WAVE_OUTPUT_FILENAME = "output.wav"
    RECORD_SECONDS = 5

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

    ''' Time to convert the audio to text '''
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
    
    print(text)
    return text

SCREEN_WIDTH = 720  # Reduzido para caber em mais telas
SCREEN_HEIGHT = int((SCREEN_WIDTH / 1429) * 1920)  # Mantém proporção
GIF_PATH = "falando.gif"

def show_gif():
    """Exibe os quadros do GIF animado no pygame."""
    pygame.init()
    screen = pygame.display.set_mode((720, 960))
    pygame.display.set_caption("Falando...")

    frames = load_gif_frames(GIF_PATH)  # Carrega os quadros do GIF
    clock = pygame.time.Clock()
    frame_index = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))
        screen.blit(frames[frame_index], (0, 0))  # Exibe o quadro atual
        pygame.display.flip()
        
        frame_index = (frame_index + 1) % len(frames)  # Avança para o próximo quadro
        clock.tick(10)  # Ajusta a velocidade da animação
        pygame.time.wait(200)

    pygame.quit()

def speak(text):
    TEXT = text
    VOICE = "pt-BR-ThalitaMultilingualNeural"
    OUTPUT = "output.mp3"

    comunicate = edge_tts.Communicate(TEXT, VOICE)
    
    with open(OUTPUT, "wb") as f:
        for chunk in comunicate.stream_sync():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
    
    sound = AudioSegment.from_mp3(OUTPUT)

    play(sound)
    
    #play_thread = threading.Thread(target=play, args=(sound,))
    #play_thread.start()

    # Aguarda um pouco até o áudio começar (~0.5s)
    #time.sleep(0.81)

    #gif_thread = threading.Thread(target=show_gif)
    #gif_thread.start()

    #play_thread.join()  # Espera o áudio terminar
    #pygame.event.post(pygame.event.Event(pygame.QUIT))  # Fecha o GIF
    #gif_thread.join()  # Espera a animação do GIF terminar

load_dotenv()

api = os.getenv("DEEPSEEK_API_KEY")

llm = ChatOpenAI(
    openai_api_key=api,  
    model="deepseek/deepseek-chat:free",  
    temperature=0.7,
    base_url="https://openrouter.ai/api/v1"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "Você é um robô assistente para idosos desenvolvido pelo Laboratório Visão Robótica e Imagem da Universidade Federal do Paraná, seu nome é Lúcia e você também é uma idosa, foi dada a seguinte instrução: {input}\n"
     "Suas respostas deverão ser claras, amigáveis e objetivas, utilizando uma linguagem simples e educada. Responda sempre em português e sem o uso de emojis, pois a sua resposta será convertida para áudio."
     "A mensagem foi enviada em {time}.")
])  

def main():
    db_handler.init_db()
    reminder = ToolReminder()

    tools = [
        Tool(
            name="Adicionar lembrete",
            func=reminder.add_reminder,
            description="Ferramenta para adicionar lembretes ao banco de dados utilizando uma descrição e uma data/hora, a estrutura deverá ser um json com as chaves 'description' e 'date_time'"
        ),
        Tool(
            name="Listar lembretes",
            func=reminder.list_reminders,
            description="Ferramenta para listar lembretes armazenados"
        ),
        Tool(
            name="Remover lembrete",
            func=reminder.remove_reminder,
            description="Ferramenta para remover lembretes armazenados utilizando uma descrição e uma data/hora, a estrutura deverá ser um json com as chaves 'description' e 'date_time'"
        )
    ]
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True,
        handle_parsing_errors=True
    )

    ''' User input'''
    user_input = "Olá, quem é você?"

    #verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
    #score = verification.verify_files("voices/Luan.wav", "voices/Klebe.wav")
    #print("Similaridade", score)

    ''' Prompt'''
    message_date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    formatted_prompt = prompt.format(input=user_input, time=message_date)

    ''' Chatbot response'''
    response = agent.invoke({"input": formatted_prompt})
    print(response["output"])

    ''' Text to speech'''
    speak(response["output"])

main()

