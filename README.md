# RAG chatbot

**Aim:** To create a RAG chatbot using an article on How July 2024 will be for the Sagittarius zodiac <br>
**Usecase:** To chat and have a conversation with the chatbot on how their July 2024 will be like. <br>
**Model used:** llama3-8b-8192 (using Groq API) <br>
**Database used:** Pinecone Vector Database <br>
**Frontend**: Streamlit <br>
**Embedding model**: Sentence Transformers <br>


## Implementation steps:

1) Setup API keys in .streamlit/secrets.toml file
2) Setup Groq API and Pinecone API for the LLM and Vector Database respectively
3) Split the article into chunks of 1000 characters
4) Generate embeddings for each chunk using Sentence Transformers
5) Store the chunks in Pinecone Vector Database
6) Created a simple Streamlit app to chat with the RAG chatbot