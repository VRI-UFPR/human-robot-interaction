import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
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
    "   - Sem emojis. \n"
    "   - Frases curtas com pausas naturais.\n"
    "   - Sempre em português brasileiro formal.\n\n"

    "Mensagem recebida: {input}\n"
    "Lembre-se: esta é a primeira impressão do usuário com tecnologia assistiva - priorize empatia e paciência!"
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"), 
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
        user_input, speaker_name = utils.get_input(calibration=True)

        ''' Chatbot response'''
        response = llm_chain.run(user_input)
        print(response)

        ''' Text to speech'''
        utils.speak(response, GIF_PATH)

if __name__ == "__main__":
    main()