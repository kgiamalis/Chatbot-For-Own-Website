import streamlit as st
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from HTMLTemplate import css, user_template, bot_template
from dotenv import load_dotenv
import os

loaders=CSVLoader('personal_posts.csv', encoding='utf-8')
docs=loaders.load()

openai_key = st.secrets["openai"]["openai_api_key"]
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["openai_api_key"]

def get_text_chunks(docs):
    text_splitter=CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    text_chunks=text_splitter.split_documents(docs)
    return text_chunks

def get_vector_store(text_chunks):
    embeddings=OpenAIEmbeddings()
    vectorstore=FAISS.from_documents(text_chunks, embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm=ChatOpenAI()
    memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain=ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain

def handle_user_input(user_question):
    if st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.warning("Please press 'Start' before asking a question.")

def main():
    load_dotenv()
    st.set_page_config(page_title="kgiamalis.co chatbot - press start button to initiate", page_icon=":chatbot:")
    st.write(css, unsafe_allow_html=True)
    st.header("kgiamalis.co chatbot 💬")
    if "conversation" not in st.session_state:
        st.session_state.conversation=None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=None
    user_question=st.text_input("Ask your question")
    if user_question:
        handle_user_input(user_question)
    with st.sidebar:
        st.title("LLM Chatapp using LangChain - Press Start to begin.")
        st.markdown('''
        This app is an LLM powered Chatbot built using:
        - [Streamlit](https://streamlit.io/) 
        - [OpenAI](https://platform.openai.com/docs/models) LLM
        - [LangChain](https://python.langchain.com/)
        ''')

        if st.button("Start"):
            with st.spinner("Processing"):
                # Load the Data
                data=docs
                #Split the Text into Chunks
                text_chunks = get_text_chunks(docs)
                print(len(text_chunks))
                #Create a Vector Store
                vectorstore=get_vector_store(text_chunks)
                #Create a Conversation Chain
                st.session_state.conversation=get_conversation_chain(vectorstore)

                st.success("Completed")

if __name__ == '__main__':
    main()