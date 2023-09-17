#Import Libraries
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

SYSTEM_PROMPT = """"
You are an assistant that provides answers based explicitly on the content of the provided dataset. 
You answers should be helpful and clear, while you should provide the URL of the blog post as well.
Examples:
- user: "What is curse of knowledge?"
assistant: "The "Curse of Knowledge" is a cognitive bias. It arises when individuals deeply versed in a topic struggle to see it from the perspective of those less informed. Essentially, once you've internalized a piece of knowledge, it becomes challenging to recall not knowing it. Here is a helpful blogpost: https://kgiamalis.co/blog/the-curse-of-knowledge"

- user: "What is a good acquisition KPI?"
assistant: "A good  acquisition KPI is Percentage of user acquisitions that signed up and converted within the same day, referred to as "single touch acquisitions." Here is a helpful blogpost: https://kgiamalis.co/blog/what-is-a-good-acquisition-kpi"

- user: "What is an important user metric?"
assistant: "Speed of platform is an important user metric. Here is a helpful blogpost: https://kgiamalis.co/blog/what-is-an-important-user-metric"
"""

#Load Data with LangChain CSVLoader
loaders=CSVLoader('personal_posts.csv', encoding='utf-8')
docs=loaders.load()

#Set OpenAI API Key
openai_key = st.secrets["openai"]["openai_api_key"]
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["openai_api_key"]

#Prepare data for embedding
def get_text_chunks(docs):
    text_splitter=CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    text_chunks=text_splitter.split_documents(docs)
    return text_chunks

#Embed the data in FAISS
def get_vector_store(text_chunks):
    embeddings=OpenAIEmbeddings()
    vectorstore=FAISS.from_documents(text_chunks, embeddings)
    return vectorstore

#Create a Conversation Chain
def get_conversation_chain(vectorstore):
    llm=ChatOpenAI(temperature=0.0)
    memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain=ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain

#Handle User Input
def handle_user_input(user_question):
    full_query = f"{SYSTEM_PROMPT} {user_question}"
    if st.session_state.conversation:
        response = st.session_state.conversation({'question': full_query})
        st.session_state.chat_history = response['chat_history']
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.warning("Please press 'Start' before asking a question.")

#Main Function
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