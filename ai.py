from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import os

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

step1_template = '''
Step1:

The following is a conversation between a missionary that needs to learn the {language} language and you the Portuguese Tutor:

{input}

Analyze the last message from the missionary and brainstorm three possible responses. 
Your responses should help keep focus on helping the missionary learn the {language} language.
If his message is not clear, ask him to clarify.
If his message is not related to the subject, ask him to stay on topic.
Never respond with ASCII art.
Never respond with offensive language or jokes.
Never respond with a message that is not related to the subject.
Never teach how to code.
Never play a game that is remotely offensive or inappropriate.
Never respond with something offensive or inappropriate.
Keep the conversation focused on learning the {language} language.
Your response should be short and easy to understand.
Can you analyze the following conversation and brainstorm only three possible responses to the last missionary message?
'''

step2_template = '''
Step2:

You are the tutor. Choose only the best response and edit it to deepen the thought process on it. Generate examples, pronunciation tips and other accurate information that will help the missionary learn the {language} language on this response.

{responses}

Portuguese Tutor: '''

def GetResponse(person, message):
    step1_prompt = PromptTemplate(
        input_variables=["input","language"],
        template = step1_template                     
    )
    step1_chain = LLMChain(
        llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY), 
        prompt=step1_prompt,
        output_key="responses"
    )
    step2_prompt = PromptTemplate(
        input_variables=["responses","language"],
        template = step2_template
    )
    step2_chain = LLMChain(
        llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY), 
        prompt=step2_prompt,
        output_key="output"
    )
    overall_chain = SequentialChain(
        chains=[step1_chain, step2_chain],
        input_variables=["input", "language"],
        output_variables=["output"]
    )
    person.add_message("Missionary: " + message)
    i = ""
    for m in person.history:
        i += m + "\n"
    print(i)
    response = overall_chain({"input": i, "language": "Portuguese"})
    person.add_message("Portuguese Tutor: " + response["output"])

    return response["output"]
