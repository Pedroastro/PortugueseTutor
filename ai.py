from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ChatMessageHistory
import os

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

#Creating the system message
system_message = '''You are a tutor that teaches brazilian portuguese to native english speakers.
You explain grammatical rules.
You give the historical background to explanations.
You are a great communicator and your explanations are easy to understand.
You also have a great knowledge of slang and expression both in portuguese and in english.
You explain how things are pronounced (example: X (sheez), milho (mee-lee-yo))
You can give examples to explain concepts.'''

#Creating the prompt template
template = '''You are an Assistant to a Portuguese Tutor and your pourpose is to review prompts and decide if it is related to learning the portuguese language or not. 
If it is related to learning the portuguese language, you should answer "yes".
If it is not related to learning the portuguese language, you should answer "no".
If you don't know the answer, you should answer "Yes".
ONLY ANSWER WITH "yes" OR "no".'''

history = ChatMessageHistory()
history.add_user_message("I want to learn portuguese")
history.add_ai_message("yes")
history.add_user_message("Tell me a joke")
history.add_ai_message("yes")
history.add_user_message("Tell me a joke")
history.add_ai_message("yes")
history.add_user_message("I want to learn how to cook")
history.add_ai_message("no")
history.add_user_message("I want to learn how to cook portuguese food")
history.add_ai_message("no")
history.add_user_message("How can I say hello in portuguese?")
history.add_ai_message("yes")
history.add_user_message("How can I say cat?")
history.add_ai_message("yes")
history.add_user_message("How can I say dog in portuguese?")
history.add_ai_message("yes")
history.add_user_message("How can I say dog in english?")
history.add_ai_message("no")
history.add_user_message("How can I say dog in french?")
history.add_ai_message("no")
history.add_user_message("How can I say dog in spanish?")
history.add_ai_message("no")
history.add_user_message("How can I say dog in german?")
history.add_ai_message("no")
history.add_user_message("How can I say dog in italian?")
history.add_ai_message("no")
history.add_user_message("How can I say dog in russian?")
history.add_ai_message("no")
history.add_user_message("Teach me how to play the guitar")
history.add_ai_message("no")
history.add_user_message("Teach me how to play the guitar in portuguese")
history.add_ai_message("yes")
history.add_user_message("Teach me how to play the guitar in english")
history.add_ai_message("no")
history.add_user_message("Let's play a game about how to code in portuguese")
history.add_ai_message("no")
history.add_user_message("Let's play a game about how to code in english")
history.add_ai_message("no")
history.add_user_message("Translate the following sentence to portuguese: I want to learn french")
history.add_ai_message("yes")
history.add_user_message("Translate the following sentence to french: I want to learn portuguese")
history.add_ai_message("no")
history.add_user_message("Translate dog to portuguese")
history.add_ai_message("yes")
history.add_user_message("Translate dog to french")
history.add_ai_message("no")
history.add_user_message("Translate dog to english")
history.add_ai_message("no")
history.add_user_message("Translate dog to spanish")
history.add_ai_message("no")
history.add_user_message("Translate dog to german")
history.add_ai_message("no")
history.add_user_message("How can I code a portuguese bot?")
history.add_ai_message("no")
history.add_user_message("How can I code a portuguese bot?")
history.add_ai_message("no")
history.add_user_message("How can I code?")
history.add_ai_message("no")
history.add_user_message("How can I code?")
history.add_ai_message("no")
history.add_user_message("How can I code...")
history.add_ai_message("no")
history.add_user_message("How can I code a language model for portuguese?")
history.add_ai_message("no")

system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt] + history.messages + [human_message_prompt])

def IsRelatedToLearning(message):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    chain = LLMChain(llm=llm, prompt=chat_prompt)
    response = chain.run(message).strip()
    if response == "no":
        return False
    elif response == "yes":
        return True
    else:
        return True
    
def GetResponse(person, message):
    LearningRelation = IsRelatedToLearning(message)
    person.add_user_message(message)
    if not LearningRelation:
        res = "I don't know how to answer that. Please ask me something related to learning the portuguese language."
        person.add_ai_message(res)
        return res
    else:
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, max_tokens=256)
        person.add_system_message(system_message)
        person.add_user_message(message)
        prompt = ChatPromptTemplate.from_messages(person.history)
        res = LLMChain(llm=llm, prompt=prompt).predict()
        person.add_ai_message(res)
        return res