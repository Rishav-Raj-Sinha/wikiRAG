import streamlit as st
import wikipedia
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.chat_models import init_chat_model
from langchain.chains import RetrievalQA
import time

st.title("WikiRAG")
# Initialize session states
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'current_search_query' not in st.session_state:
    st.session_state.current_search_query = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


def get_wikipedia_link(search_term):
    try:
        page_title = wikipedia.search(search_term)[0]  # Get the first search result
        page_url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
        return page_url
    except Exception as e:
        return f"Error: {e}"

search_query = st.text_input("enter search terms")
if search_query and search_query != st.session_state.current_search_query:

    wiki_link = get_wikipedia_link(search_query)
    st.write(wiki_link)
    loader = WebBaseLoader(wiki_link)
    docs = loader.load()
    chunk_size =500
    chunk_overlap = 100
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function = len
    )
    chunks = r_splitter.split_documents(docs)
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",
        google_api_key=st.secrets["GOOGLE_API_KEY"])
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("wikrag")
    index.delete(delete_all=True)
    vector_store = PineconeVectorStore(embedding=embeddings, index=index)
    document_ids = vector_store.add_documents(documents=chunks)
    time.sleep(5)
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    llm = init_chat_model("llama3-8b-8192", model_provider="groq")
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_store.as_retriever()
    )
    # Store in session state
    st.session_state.vector_store = vector_store
    st.session_state.qa_chain = qa_chain
    st.session_state.current_search_query = search_query
    st.session_state.processing_complete = True  # Set flag to True when processing is complete
    st.success('Ready for questions!')

if st.session_state.processing_complete:
    question = st.chat_input("enter your query")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if question and st.session_state.qa_chain is not None:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("assistant"):
            result = st.session_state.qa_chain({"query":question})
            st.markdown(result["result"])
        st.session_state.messages.append({"role": "assistant", "content": result["result"]})
