import streamlit as st
from langchain_groq import ChatGroq

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores import Pinecone
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

import os

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from langchain import PromptTemplate

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

import pinecone

#test comment

# Initialize Streamlit app
st.title("RAG Chatbot")


# Load API keys from Streamlit secrets
groq_api_key = st.secrets["general"]["groq_api_key"]
pinecone_api_key = st.secrets["general"]["pinecone_api_key"]

os.environ['PINECONE_API_KEY'] = pinecone_api_key


# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)

# Load and process documents
# loader = TextLoader('./data.txt')
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
# docs = text_splitter.split_documents(documents)

@st.cache_resource
def get_docsearch():
    embeddings = HuggingFaceEmbeddings()
    #pc = Pinecone(api_key=pinecone_api_key)
    index_name = "rag-index"
    
    with open("output.txt", "r") as file:
        output_text = file.read()
    
    return PineconeVectorStore.from_texts(texts=output_text,embedding=embeddings, index_name=index_name)

# Initialize Pinecone

# if index_name not in pc.list_indexes().names():
#   pc.create_index(name=index_name, metric="cosine", dimension=768, spec=ServerlessSpec(
#             cloud='aws', 
#             region='us-east-1'
#         ))
#   docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name) #docsearch stores the embeddings of the docs
# else:



# Define prompt template
template = """
You are a fortune teller. This human will ask you questions about their life. 
Use following piece of context to answer the question. 
If you don't know the answer, just say you don't know. 
Keep the answer within 2 sentences and concise.

Context: {context}
Question: {question}
Answer: 
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Define RAG chain
@st.cache_resource
def get_rag_chain():
    llm = get_llm()
    docsearch = get_docsearch()
    return (
        {"context": docsearch.as_retriever(), "question": RunnablePassthrough()} 
        | prompt 
        | llm
        | StrOutputParser()
    )

rag_chain = get_rag_chain()

# Streamlit input and output
user_input = st.text_input("As a person of Sagittarius zodiac, ask anything about how your July 2024 be:")
if st.button("Generate answer"):
    if user_input:
        with st.spinner("Generating answer..."):
            result = rag_chain.invoke(user_input)
        st.write(result)
    else:
        st.write("Please enter a question.")