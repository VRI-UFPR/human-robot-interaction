import os
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain_core.prompts import ChatPromptTemplate
import edge_tts
from pydub import AudioSegment
from pydub.playback import play
import pygame
import threading
from PIL import Image
import time
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
import pyaudio
import wave
import subprocess
import re


SCREEN_WIDTH = 720  # Reduzido para caber em mais telas
SCREEN_HEIGHT = int((SCREEN_WIDTH / 1429) * 1920)  # Mantém proporção
GIF_PATH = "falando.gif"

load_dotenv()

api = os.getenv("DEEPSEEK_API_KEY")

llm = ChatOpenAI(
    openai_api_key=api,  
    model="deepseek/deepseek-chat:free",  
    temperature=0.7,
    base_url="https://openrouter.ai/api/v1"
)

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


def get_input(calibration = False):
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

    ''' If there's a calibration, the user will say "Meu nome é [Nome]" '''
    if 'nome é' in text:
        calibration = True

    ''' If calibration copy the audio in a folder called voices with the name of the user '''
    if calibration:
        match = re.search(r"nome é (\w+)", text)
        name = match.group(1)
        os.makedirs(f"voices/", exist_ok=True)
        os.rename(WAVE_OUTPUT_FILENAME, f"voices/{name}.wav")
        
    return text

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
    
    play_thread = threading.Thread(target=play, args=(sound,))
    play_thread.start()

    # Aguarda um pouco até o áudio começar (~0.5s)
    time.sleep(0.81)

    gif_thread = threading.Thread(target=show_gif)
    gif_thread.start()

    play_thread.join()  # Espera o áudio terminar
    pygame.event.post(pygame.event.Event(pygame.QUIT))  # Fecha o GIF
    gif_thread.join()  # Espera a animação do GIF terminar

prompt = ChatPromptTemplate.from_messages([
    ("system", 
    "Você é a Lúcia, um robô assistente para idosos desenvolvido pelo Laboratório Visão Robótica e Imagem da UFPR. Sua personalidade é de uma idosa gentil e paciente.\n"
    "Neste momento, você está realizando a primeira interação com o usuário e precisa calibrar sua voz seguindo estes passos:\n\n"

    "1. **SAUDAÇÃO INICIAL:**\n"
    "   - Cumprimente calorosamente, apresente-se brevemente e explique o propósito da calibração.\n"
    "   - Exemplo: 'Olá! Sou a Lúcia, sua nova companheira digital da Universidade Federal do Paraná. "
    "Antes de começarmos, me ajude a reconhecer sua voz para que eu possa identificar você claramente!'\n\n"

    "2. **INSTRUÇÕES DE CALIBRAÇÃO:**\n"
    "   - Peça para o usuário repetir uma frase específica (exemplo: 'Bom dia! Meu nome é [Nome]. Vamos conversar?').\n"
    "   - Oriente que ele fale pausadamente e em volume natural.\n"
    "   - Destaque que isso garantirá uma calibração melhor.\n\n"

    "3. **FEEDBACK:**\n"
    "   - Após a tentativa, analise a entrada do usuário.\n"
    "   - Se bem-sucedido: confirme e agradeça, finalize a etapa de calibração e finalize a interação.\n\n"

    "**FORMATO GERAL:**\n"
    "   - Linguagem simples, afetuosa e encorajadora.\n"
    "   - Evite termos técnicos.\n"
    "   - Frases curtas com pausas naturais.\n"
    "   - Sempre em português brasileiro formal.\n\n"

    "Mensagem recebida: {input}\n"
    "Lembre-se: esta é a primeira impressão do usuário com tecnologia assistiva - priorize empatia e paciência!"
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

def main():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True    
    )

    ''' Repeats until CTRL+C is pressed '''
    
    while True:
        user_input = get_input()

        ''' Chatbot response'''
        response = llm_chain.run(user_input)
        print(response)

        ''' Text to speech'''
        #speak(response["output"])


main()