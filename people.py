class Person:
    def __init__(self, id):
       self.id = id
       self.history = []
       self.ai_responses = []

    def add_message(self, message):
        self.history.append(message)