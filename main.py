from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import db_handler
from tool_reminder import ToolReminder
from langchain.agents import AgentType, initialize_agent
from langchain.agents import Tool
import datetime
import os
import utils
    
GIF_PATH = "falando.gif"

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
     "A mensagem foi enviada em {time}."
     " Caso, o nome do usuário seja reconhecido, utilize-o para se referir a ele."
     "Nome usuário: {speaker_name}")
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
    user_input, speaker_name = utils.get_input()

    ''' Prompt'''
    message_date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    formatted_prompt = prompt.format(input=user_input, time=message_date, speaker_name=speaker_name)

    ''' Chatbot response'''
    response = agent.invoke({"input": formatted_prompt})
    print(response["output"])

    ''' Text to speech'''
    utils.speak(response["output"], GIF_PATH)

if __name__ == "__main__":
    main()
