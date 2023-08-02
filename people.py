from langchain.schema.messages import SystemMessage, AIMessage, HumanMessage

class Person:
    def __init__(self, id):
       self.id = id
       self.history = []
    
    def add_user_message(self, message):
        self.history.append(HumanMessage(content=message))

    def add_ai_message(self, message):
        self.history.append(AIMessage(content=message))
    
    def add_system_message(self, message):
        self.history.append(SystemMessage(content=message))