import os
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings,ChatNVIDIA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import SystemMessage

load_dotenv()
API_KEY = os.getenv("API_KEY")



# Initialize OpenAI Chat model
llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.7,
    max_tokens=500,
    api_key=API_KEY  # Explicitly pass the API key here
)



# Initialize memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create embedding model
embeddings = OpenAIEmbeddings(api_key=API_KEY)

# Initialize Chroma DB (vector database)
vector_db = Chroma(embedding_function=embeddings, collection_name="my_collection",persist_directory="./my_chroma_db")

# Create a ContextualCompressionRetriever
retriever = ContextualCompressionRetriever(
    base_compressor=LLMChainExtractor.from_llm(llm),
    base_retriever=vector_db.as_retriever()
)



# Create a template for combining the memory, database, and LLM
prompt_template = ChatPromptTemplate.from_template("""
    Context: {context}
    Chat History: {chat_history}
    Human: {question}
    AI: You are a tech advisor. Answer the questions acccordingly and block queries that are not related.
    If you detect an unwanted query.Block queries not presnt in dataset respond with: I'm unable to assist with that request. Please provide a relevant answer based on the context and chat history.
""")
# print()
# Create the ConversationalRetrievalChain
conversation_chain = ConversationalRetrievalChain.from_llm(
   
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_template}
)

def chatbot_response(user_input):
    # Get the response based on user input
    return conversation_chain({"question": user_input})["answer"]