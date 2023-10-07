from flask import Flask, request
from langchain.memory import ConversationBufferMemory
import requests
import os
import json
import csv
from people import Person
import LanguageTutor
import EvangelhoGPT
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

VERIFY_TOKEN = os.getenv('VERIFY_TOKEN')
PAGE_ACCESS_TOKEN = os.getenv('MESSENGER_API_TOKEN')
LANGUAGE_TUTOR_ID = os.getenv('LANGUAGE_TUTOR_ID')
EVANGELHOGPT_ID = os.getenv('EVANGELHOGPT_ID')
LANGUAGE_TUTOR_PHONE=os.getenv('LANGUAGE_TUTOR_PHONE')
EVANGELHOGPT_PHONE=os.getenv('EVANGELHOGPT_PHONE')

language_tutor_people = []
evangelhogpt_people = []

def send_message(recipient_id, text, pageid):
    """Send a response to Facebook"""
    payload = {
      "messaging_product": "whatsapp",
      "recipient_type": "individual",
      "to": recipient_id,
      "type": "text",
      "text": {
        "preview_url": False,
        "body": text
        }
    }
    data = json.dumps(payload)

    response = requests.post(
        f'https://graph.facebook.com/v17.0/{pageid}/messages', 
        data=data, headers={'Authorization': 'Bearer ' + PAGE_ACCESS_TOKEN, 'Content-Type': 'application/json'}, timeout=20)

    if response.status_code == 200:
        return response.json()
    else:
        return None

def verify_webhook(req):
    if req.args.get("hub.verify_token") == VERIFY_TOKEN:
        return req.args.get("hub.challenge")
    else:
        return "incorrect"

def does_person_exist(id, people_array):
    for person in people_array:
        if person.id == id:
            return True
    return False

def get_person(id, people_array):
    for person in people_array:
        if person.id == id:
            return person
    return None

def is_mission_phone(phone):
    with open('phone_number.csv', 'r') as list:
        reader = csv.reader(list)
        for row in reader:
            if phone == row[0]:
                return True
        return False

def language_tutor_response(sender, message):
    """Formulate a response to the user and
    pass it on to a function that sends it."""

    if not does_person_exist(sender, language_tutor_people):
        language_tutor_people.append(Person(sender))
        send_message(sender, '''Hi! I'm your personal language assistant. I can help you practice your Portuguese. What would you like to know?
        /portugues-ingles (falo português e quero aprender inglês)
        /espanhol-portugues (falo espanhol e quero aprender português)
        /portugues-espanhol (falo português e quero aprender espanhol)
        /ingles-portugues (falo inglês e quero aprender português - configuração padrão)
        /limpar (limpar a memória do chat)''', LANGUAGE_TUTOR_ID)
    else:
        if get_person(sender, language_tutor_people).message_count >= 21:
            response = "Você atingiu o limite de mensagens diárias"
        elif message[0] == "/":
            if message == '/limpar':
                get_person(sender, language_tutor_people).history = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
                response = 'Chat limpo'
            elif message == '/portugues-ingles' or message == 'portugues-ingles':
                get_person(sender, language_tutor_people).change_language("portugues-ingles")
                response = 'Olá! Sou seu assistente pessoal de idiomas. Posso ajudá-lo a praticar o seu inglês. O que você gostaria de saber?'
            elif message == '/espanhol-portugues' or message == 'espanhol-portugues':
                get_person(sender, language_tutor_people).change_language("espanhol-portugues")
                response = '¡Hola! Soy tu asistente personal de idiomas. ¿Puedo ayudarte a practicar tu portugués? ¿Qué te gustaría saber?'
            elif message == '/portugues-espanhol' or message == 'portugues-espanhol':
                get_person(sender, language_tutor_people).change_language("portugues-espanhol")
                response = 'Olá! Sou seu assistente pessoal de idiomas. Posso ajudá-lo a praticar o seu espanhol. O que você gostaria de saber?'
            elif message == '/ingles-portugues' or message == 'ingles-portugues':
                get_person(sender, language_tutor_people).change_language("ingles-portugues")
                response = "Hello! I'm your personal language assistant. Can I help you practice your Portuguese? What would you like to know?"
        else:
            response = LanguageTutor.get_response(get_person(sender, language_tutor_people), message)
            get_person(sender, language_tutor_people).message_count += 1
        send_message(sender, response, LANGUAGE_TUTOR_ID)

def evangelhogpt_response(sender, message):
    if not does_person_exist(sender, evangelhogpt_people):
        evangelhogpt_people.append(Person(sender))
        send_message(sender, '''Sou um assistente feito para te ajudar com perguntas sobre o evangelho ou perguntas sobre a obra missionária.''', EVANGELHOGPT_ID)
    else:
        if get_person(sender, evangelhogpt_people).message_count >= 21:
            response = "Você atingiu o limite de mensagens diárias"
        else:
            response = EvangelhoGPT.get_response(get_person(sender, evangelhogpt_people), message)
        send_message(sender, response, EVANGELHOGPT_ID)

@app.route("/", methods=['GET', 'POST'])
def listen():
    """This is the main function flask uses to 
    listen at the `/webhook` endpoint"""
    if request.method == 'GET':
        return verify_webhook(request)
    
    elif request.method == 'POST':
        payload = request.json
        print(payload)
        try:
            event = payload['entry'][0]['changes'][0]['value']['messages']
            recipient = payload['entry'][0]['changes'][0]['value']['metadata']['display_phone_number']
        except KeyError:
            return "ok"
        for x in event:
            try:
                text = x['text']['body']
                sender_id = x['from']
            except KeyError:
                return "ok"
            if recipient == LANGUAGE_TUTOR_PHONE:
                language_tutor_response(sender_id, text)
            elif recipient == EVANGELHOGPT_PHONE:
                evangelhogpt_response(sender_id, text)
        return "ok"
    
if __name__ == "__main__":
    app.run()