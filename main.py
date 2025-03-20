import requests
import utils
    
GIF_PATH = "falando.gif"

URL = "http://127.0.0.1:8080/processing"

def main():
    
    ''' User input'''
    user_input, speaker_name = utils.get_input()

    ''' Send request '''
    payload = {"user_input":user_input, "speaker_name": speaker_name}
    response = requests.post(URL, json=payload)

    response_text = response.json()["processed_text"] 
   
    ''' Text to speech'''
    utils.speak(response_text, GIF_PATH)

if __name__ == "__main__":
    main()
