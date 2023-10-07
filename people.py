from langchain.memory import ConversationBufferMemory

class Person:
    def __init__(self, id):
       self.id = id
       self.history = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
       self.language = 'ingles-portugues'
       self.message_count = 0
    
    def change_language(self, language):
        self.language = language