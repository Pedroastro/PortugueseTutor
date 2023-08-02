from flask import Flask, request
import requests
import os
import json
from people import Person
from ai import GetResponse

app = Flask(__name__)

VERIFY_TOKEN = os.getenv('VERIFY_TOKEN') # paste your verify token here
PAGE_ACCESS_TOKEN = os.getenv('MESSENGER_API_TOKEN') # paste your page access token here
PAGE_ID = os.getenv('PAGE_ID') # paste your page id here
PASSWORD= os.getenv('PASSWORD') # paste your bot name here

people = []


def send_message(recipient_id, text):
    """Send a response to Facebook"""
    payload = {
        "recipient": {
            "id": recipient_id
        },
        "messaging_type": "RESPONSE",
        "message": {
            "text": text
        }
    }
    data = json.dumps(payload)

    response = requests.post(
        f'https://graph.facebook.com/v17.0/{PAGE_ID}/messages?access_token={PAGE_ACCESS_TOKEN}', 
        data=data, headers={'Content-Type': 'application/json'})

    if response.status_code == 200:
        return response.json()
    else:
        return None

def verify_password(password):
    if password == PASSWORD:
        return True
    else:
        return False

def verify_webhook(req):
    if req.args.get("hub.verify_token") == VERIFY_TOKEN:
        return req.args.get("hub.challenge")
    else:
        return "incorrect"

def respond(sender, message):
    """Formulate a response to the user and
    pass it on to a function that sends it."""

    if not DoesPersonExist(sender):
        if verify_password(message):
            people.append(Person(sender))
            send_message(sender, "Hi! I'm your personal language coach. I can help you practice your Portuguese. What would you like to know?")
        else:
            send_message(sender, "Please enter the correct password to get my help ;)")
    else:
        if message == "quit":
            people.remove(GetPerson(sender))
        else:
            person = GetPerson(sender)
            response = GetResponse(person, message)
            send_message(sender, response)

def DoesPersonExist(id):
    for person in people:
        if person.id == id:
            return True
    return False

def GetPerson(id):
    for person in people:
        if person.id == id:
            return person
    return None

@app.route("/", methods=['GET', 'POST'])
def listen():
    """This is the main function flask uses to 
    listen at the `/webhook` endpoint"""
    if request.method == 'GET':
        return verify_webhook(request)
    
    elif request.method == 'POST':
        payload = request.json
        print(payload)
        event = payload['entry'][0]['messaging']
        for x in event:
            text = x['message']['text']
            sender_id = x['sender']['id']
            respond(sender_id, text)

        return "ok"
    
if __name__ == "__main__":
    app.run()