import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.document_transformers import LongContextReorder
from langchain.agents import Tool
from langchain.agents import initialize_agent
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DB_FAISS_PATH = 'Vectorstores/'

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2",
                                       model_kwargs={'device': 'cpu'})

def search_vectorstore(query, db):
    """Search the vector store for relevant documents of the Gospel Library from the Church of Jesus Christ of Latter Day Saints"""
    retriever = db.as_retriever(search_kwargs={'k': 2})
    docs = retriever.get_relevant_documents(query)
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)
    return reordered_docs

def search_adultos(query):
    db = FAISS.load_local(DB_FAISS_PATH + 'Adultos', embeddings)
    return search_vectorstore(query, db)

def search_ajuda_para_a_vida(query):
    db = FAISS.load_local(DB_FAISS_PATH + 'Ajuda para a vida', embeddings)
    return search_vectorstore(query, db)

def search_compartilhar_o_evangelho(query):
    db = FAISS.load_local(DB_FAISS_PATH + 'Compartilhar o evangelho', embeddings)
    return search_vectorstore(query, db)

def search_conferencia_geral(query):
    db = FAISS.load_local(DB_FAISS_PATH + 'Conferencia Geral', embeddings)
    return search_vectorstore(query, db)

def search_criancas(query):
    db = FAISS.load_local(DB_FAISS_PATH + 'Crianças', embeddings)
    return search_vectorstore(query, db)

def search_escrituras(query):
    db = FAISS.load_local(DB_FAISS_PATH + 'Escrituras', embeddings)
    return search_vectorstore(query, db)

def search_historia_da_igreja(query):
    db = FAISS.load_local(DB_FAISS_PATH + 'Historia da igreja', embeddings)
    return search_vectorstore(query, db)

def search_jesus_cristo(query):
    db = FAISS.load_local(DB_FAISS_PATH + 'Jesus Cristo', embeddings)
    return search_vectorstore(query, db)

def search_jovens(query):
    db = FAISS.load_local(DB_FAISS_PATH + 'Jovens', embeddings)
    return search_vectorstore(query, db)

def search_livros_e_licoes(query):
    db = FAISS.load_local(DB_FAISS_PATH + 'Livros e licoes', embeddings)
    return search_vectorstore(query, db)

def search_manuais_e_chamados(query):
    db = FAISS.load_local(DB_FAISS_PATH + 'Manuais e Chamados', embeddings)
    return search_vectorstore(query, db)

def search_musica(query):
    db = FAISS.load_local(DB_FAISS_PATH + 'Musica', embeddings)
    return search_vectorstore(query, db)

def search_revistas(query):
    db = FAISS.load_local(DB_FAISS_PATH + 'Revistas', embeddings)
    return search_vectorstore(query, db)

def search_templo_e_historia_da_familia(query):
    db = FAISS.load_local(DB_FAISS_PATH + 'Templo e historia da familia', embeddings)
    return search_vectorstore(query, db)

def search_topicos_e_perguntas(query):
    db = FAISS.load_local(DB_FAISS_PATH + 'Topicos e Perguntas', embeddings)
    return search_vectorstore(query, db)

def search_vem_e_segue_me(query):
    db = FAISS.load_local(DB_FAISS_PATH + 'Vem e Segueme', embeddings)
    return search_vectorstore(query, db)

def search_videos_e_imagens(query):
    db = FAISS.load_local(DB_FAISS_PATH + 'Videos e imagens', embeddings)
    return search_vectorstore(query, db)

tools = [
    Tool.from_function(
    func=search_adultos,
    name="Procurar conteúdo direcionado a membros adultos da igreja",
    description = "Utilize esta função para pesquisar conteúdo direcionado a membros adultos da igreja no banco de dados da loja de vetores otimizada para documentos armazenados na Biblioteca do Evangelho da Igreja de Jesus Cristo dos Santos dos Últimos Dias, juntamente com seus embeddings, buscando documentos mais relevantes para uma consulta específica, ou seja, aqueles cujos embeddings são mais semelhantes ao embedding da consulta. A entrada para esta função deve ser uma string. Utilize-a sempre que precisar de mais conteúdo direcionado a membros adultos da igreja de dentro da Biblioteca do Evangelho. Faça pesquisas específicas para ajustar o resultado de uma forma melhor.",
    ),
    Tool.from_function(
    func=search_ajuda_para_a_vida,
    name="Procurar conteúdo relacionado a ajuda para a vida",
    description = "Utilize esta função para pesquisar conteúdo relacionado a ajuda para a vida na loja de vetores otimizada para documentos armazenados na Biblioteca do Evangelho da Igreja de Jesus Cristo dos Santos dos Últimos Dias, juntamente com seus embeddings, buscando documentos mais relevantes para uma consulta específica, ou seja, aqueles cujos embeddings são mais semelhantes ao embedding da consulta. A entrada para esta função deve ser uma string. Utilize-a sempre que precisar de mais conteúdo relacionado a ajuda para a vida da Biblioteca do Evangelho. Faça pesquisas específicas para ajustar o resultado de uma forma melhor.",
    ),
    Tool.from_function(
    func=search_compartilhar_o_evangelho,
    name="Procurar conteúdo relacionado a compartilhar o evangelho",
    description = "Utilize esta função para pesquisar conteúdo relacionado a compartilhar o evangelho na loja de vetores otimizada para documentos armazenados na Biblioteca do Evangelho da Igreja de Jesus Cristo dos Santos dos Últimos Dias, juntamente com seus embeddings, buscando documentos mais relevantes para uma consulta específica, ou seja, aqueles cujos embeddings são mais semelhantes ao embedding da consulta. A entrada para esta função deve ser uma string. Utilize-a sempre que precisar de mais conteúdo relacionado a compartilhar o evangelho da Biblioteca do Evangelho. Faça pesquisas específicas para ajustar o resultado de uma forma melhor.",
    ),
    Tool.from_function(
    func=search_conferencia_geral,
    name="Procurar conteúdo relacionado a conferência geral",
    description = "Utilize esta função para pesquisar conteúdo relacionado a conferência geral na loja de vetores otimizada para documentos armazenados na Biblioteca do Evangelho da Igreja de Jesus Cristo dos Santos dos Últimos Dias, juntamente com seus embeddings, buscando documentos mais relevantes para uma consulta específica, ou seja, aqueles cujos embeddings são mais semelhantes ao embedding da consulta. A entrada para esta função deve ser uma string. Utilize-a sempre que precisar de mais conteúdo relacionado a conferência geral da Biblioteca do Evangelho. Faça pesquisas específicas para ajustar o resultado de uma forma melhor.",
    ),
    Tool.from_function(
    func=search_criancas,
    name="Procurar conteúdo direcionado a crianças",
    description = "Utilize esta função para pesquisar conteúdo direcionado a crianças na loja de vetores otimizada para documentos armazenados na Biblioteca do Evangelho da Igreja de Jesus Cristo dos Santos dos Últimos Dias, juntamente com seus embeddings, buscando documentos mais relevantes para uma consulta específica, ou seja, aqueles cujos embeddings são mais semelhantes ao embedding da consulta. A entrada para esta função deve ser uma string. Utilize-a sempre que precisar de mais conteúdo direcionado a crianças da Biblioteca do Evangelho. Faça pesquisas específicas para ajustar o resultado de uma forma melhor.",
    ),
    Tool.from_function(
    func=search_escrituras,
    name="Procurar escrituras",
    description = "Utilize esta função para pesquisar escrituras na loja de vetores otimizada para documentos armazenados na Biblioteca do Evangelho da Igreja de Jesus Cristo dos Santos dos Últimos Dias, juntamente com seus embeddings, buscando documentos mais relevantes para uma consulta específica, ou seja, aqueles cujos embeddings são mais semelhantes ao embedding da consulta. A entrada para esta função deve ser uma string. Utilize-a sempre que precisar de mais escrituras da Biblioteca do Evangelho. Faça pesquisas específicas para ajustar o resultado de uma forma melhor.",
    ),
    Tool.from_function(
    func=search_historia_da_igreja,
    name="Procurar conteúdo relacionado a história da igreja",
    description = "Utilize esta função para pesquisar conteúdo relacionado a história da igreja na loja de vetores otimizada para documentos armazenados na Biblioteca do Evangelho da Igreja de Jesus Cristo dos Santos dos Últimos Dias, juntamente com seus embeddings, buscando documentos mais relevantes para uma consulta específica, ou seja, aqueles cujos embeddings são mais semelhantes ao embedding da consulta. A entrada para esta função deve ser uma string. Utilize-a sempre que precisar de mais conteúdo relacionado a história da igreja na Biblioteca do Evangelho. Faça pesquisas específicas para ajustar o resultado de uma forma melhor.",
    ),
    Tool.from_function(
    func=search_jesus_cristo,
    name="Procurar conteúdo sobre Jesus Cristo",
    description = "Utilize esta função para pesquisar conteúdo sobre Jesus Cristo na loja de vetores otimizada para documentos armazenados na Biblioteca do Evangelho da Igreja de Jesus Cristo dos Santos dos Últimos Dias, juntamente com seus embeddings, buscando documentos mais relevantes para uma consulta específica, ou seja, aqueles cujos embeddings são mais semelhantes ao embedding da consulta. A entrada para esta função deve ser uma string. Utilize-a sempre que precisar de mais conteúdo sobre Jesus Cristo na Biblioteca do Evangelho. Faça pesquisas específicas para ajustar o resultado de uma forma melhor.",
    ),
    Tool.from_function(
    func=search_jovens,
    name="Procurar conteúdo direcionado a jovens",
    description = "Utilize esta função para pesquisar conteúdo direcionado a jovens na loja de vetores otimizada para documentos armazenados na Biblioteca do Evangelho da Igreja de Jesus Cristo dos Santos dos Últimos Dias, juntamente com seus embeddings, buscando documentos mais relevantes para uma consulta específica, ou seja, aqueles cujos embeddings são mais semelhantes ao embedding da consulta. A entrada para esta função deve ser uma string. Utilize-a sempre que precisar de mais conteúdo direcionado a jovens na Biblioteca do Evangelho. Faça pesquisas específicas para ajustar o resultado de uma forma melhor.",
    ),
    Tool.from_function(
    func=search_livros_e_licoes,
    name="Procurar conteúdo de livros e lições da igreja",
    description = "Utilize esta função para pesquisar conteúdo de livros e lições da igreja na loja de vetores otimizada para documentos armazenados na Biblioteca do Evangelho da Igreja de Jesus Cristo dos Santos dos Últimos Dias, juntamente com seus embeddings, buscando documentos mais relevantes para uma consulta específica, ou seja, aqueles cujos embeddings são mais semelhantes ao embedding da consulta. A entrada para esta função deve ser uma string. Utilize-a sempre que precisar de mais conteúdo de livros e lições da igreja na Biblioteca do Evangelho. Faça pesquisas específicas para ajustar o resultado de uma forma melhor.",
    ),
    Tool.from_function(
    func=search_manuais_e_chamados,
    name="Procurar conteúdo de manuais e conteúdo para chamados específicos",
    description = "Utilize esta função para pesquisar conteúdo de manuais e conteúdo para chamados específicos na loja de vetores otimizada para documentos armazenados na Biblioteca do Evangelho da Igreja de Jesus Cristo dos Santos dos Últimos Dias, juntamente com seus embeddings, buscando documentos mais relevantes para uma consulta específica, ou seja, aqueles cujos embeddings são mais semelhantes ao embedding da consulta. A entrada para esta função deve ser uma string. Utilize-a sempre que precisar de mais conteúdo de manuais e conteúdo para chamados específicos na Biblioteca do Evangelho. Faça pesquisas específicas para ajustar o resultado de uma forma melhor.",
    ),
    Tool.from_function(
    func=search_musica,
    name="Procurar conteúdo de música da igreja",
    description = "Utilize esta função para pesquisar conteúdo de música da igreja na loja de vetores otimizada para documentos armazenados na Biblioteca do Evangelho da Igreja de Jesus Cristo dos Santos dos Últimos Dias, juntamente com seus embeddings, buscando documentos mais relevantes para uma consulta específica, ou seja, aqueles cujos embeddings são mais semelhantes ao embedding da consulta. A entrada para esta função deve ser uma string. Utilize-a sempre que precisar de mais conteúdo de música da igreja na Biblioteca do Evangelho. Faça pesquisas específicas para ajustar o resultado de uma forma melhor.",
    ),
    Tool.from_function(
    func=search_revistas,
    name="Procurar conteúdo de revistas igreja",
    description = "Utilize esta função para pesquisar conteúdo de revistas igreja na loja de vetores otimizada para documentos armazenados na Biblioteca do Evangelho da Igreja de Jesus Cristo dos Santos dos Últimos Dias, juntamente com seus embeddings, buscando documentos mais relevantes para uma consulta específica, ou seja, aqueles cujos embeddings são mais semelhantes ao embedding da consulta. A entrada para esta função deve ser uma string. Utilize-a sempre que precisar de mais conteúdo de revistas igreja na Biblioteca do Evangelho. Faça pesquisas específicas para ajustar o resultado de uma forma melhor.",
    ),
    Tool.from_function(
    func=search_templo_e_historia_da_familia,
    name="Procurar conteúdo relacionado a templo e história da família",
    description = "Utilize esta função para pesquisar conteúdo relacionado a templo e história da família na loja de vetores otimizada para documentos armazenados na Biblioteca do Evangelho da Igreja de Jesus Cristo dos Santos dos Últimos Dias, juntamente com seus embeddings, buscando documentos mais relevantes para uma consulta específica, ou seja, aqueles cujos embeddings são mais semelhantes ao embedding da consulta. A entrada para esta função deve ser uma string. Utilize-a sempre que precisar de mais conteúdo relacionado a templo e história da família na Biblioteca do Evangelho. Faça pesquisas específicas para ajustar o resultado de uma forma melhor.",
    ),
    Tool.from_function(
    func=search_topicos_e_perguntas,
    name="Procurar tópicos e perguntas",
    description = "Utilize esta função para pesquisar tópicos e perguntas na loja de vetores otimizada para documentos armazenados na Biblioteca do Evangelho da Igreja de Jesus Cristo dos Santos dos Últimos Dias, juntamente com seus embeddings, buscando documentos mais relevantes para uma consulta específica, ou seja, aqueles cujos embeddings são mais semelhantes ao embedding da consulta. A entrada para esta função deve ser uma string. Utilize-a sempre que precisar de mais tópicos e perguntas na Biblioteca do Evangelho. Faça pesquisas específicas para ajustar o resultado de uma forma melhor.",
    ),
    Tool.from_function(
    func=search_vem_e_segue_me,
    name='Procurar conteúdo dos manuais "Vem, e Segue-me"',
    description = 'Utilize esta função para pesquisar conteúdo dos manuais "Vem, e Segue-me" na loja de vetores otimizada para documentos armazenados na Biblioteca do Evangelho da Igreja de Jesus Cristo dos Santos dos Últimos Dias, juntamente com seus embeddings, buscando documentos mais relevantes para uma consulta específica, ou seja, aqueles cujos embeddings são mais semelhantes ao embedding da consulta. A entrada para esta função deve ser uma string. Utilize-a sempre que precisar de mais conteúdo dos manuais "Vem, e Segue-me" na Biblioteca do Evangelho. Faça pesquisas específicas para ajustar o resultado de uma forma melhor.',
    ),
    Tool.from_function(
    func=search_videos_e_imagens,
    name='Procurar conteúdo relacionado a vídeos e imagens',
    description = 'Utilize esta função para pesquisar conteúdo relacionado a vídeos e imagens na loja de vetores otimizada para documentos armazenados na Biblioteca do Evangelho da Igreja de Jesus Cristo dos Santos dos Últimos Dias, juntamente com seus embeddings, buscando documentos mais relevantes para uma consulta específica, ou seja, aqueles cujos embeddings são mais semelhantes ao embedding da consulta. A entrada para esta função deve ser uma string. Utilize-a sempre que precisar de mais conteúdo relacionado a vídeos e imagens na Biblioteca do Evangelho. Faça pesquisas específicas para ajustar o resultado de uma forma melhor.',
    )
]

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, max_retries=4, model='gpt-3.5-turbo-16k')

prompt_template = '''Você é um assistente para membros e missionários da Igreja de Jesus Cristo dos Santos dos Últimos Dias. 
Se não souber responder, somente responda que não sabe.
Se o missionário pedir uma escritura, responda com uma escritura.
NÃO MODIFIQUE OU CRIE ESCRITURAS OU CITAÇÕES, MANTENHA O TEXTO ORIGINAL.
TAMBÉM NÃO CRIE PARTES DE ESCRITURAS.
DE CITAÇÕES OU ESCRITURAS SOMENTE RETORNE O TEXTO.
Para a referência do capítulo da escritura ou citação utilizada coloque o título do arquivo utilizado como referência. 
Faça o melhor para manter a numeração dos versículos correta de acordo com o arquivo.
Se necessário pesquise várias vezes na Biblioteca do Evangelho.
Responda ao seguinte pedido ou pergunta baseando-se no conteúdo da Biblioteca do Evangelho, mas NÃO RESPONDA se o pedido ou pergunta não for relacionado ao aprendizado do evangelho:
'''
def get_response(person, message):
    conversational_agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=5,
    memory=person.history
    )
    response = conversational_agent.run(prompt_template + message)
    return response