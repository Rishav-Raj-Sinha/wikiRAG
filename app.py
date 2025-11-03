import streamlit as st
import os
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.chat_models import init_chat_model
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document

st.set_page_config(page_title="Wikipedia RAG Chatbot", page_icon="🤖", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = ""
if "pinecone_key" not in st.session_state:
    st.session_state.pinecone_key = ""


def initialize_rag_system(gemini_key, pinecone_key):
    """Initialize embeddings, vector store, and LLM"""
    try:
        # Setup embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", google_api_key=gemini_key
        )

        # Setup Pinecone
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index("wikirag")
        vector_store = PineconeVectorStore(embedding=embeddings, index=index)

        # Setup LLM
        llm = init_chat_model(
            "gemini-2.0-flash-exp", model_provider="google_genai", api_key=gemini_key
        )

        # Setup prompt and graph
        prompt = hub.pull("rlm/rag-prompt")

        class State(TypedDict):
            question: str
            context: List[Document]
            answer: str

        def retrieve(state: State):
            retrieved_docs = vector_store.similarity_search(state["question"])
            return {"context": retrieved_docs}

        def generate(state: State):
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = prompt.invoke(
                {"question": state["question"], "context": docs_content}
            )
            response = llm.invoke(messages)
            return {"answer": response.content}

        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()

        return graph, vector_store
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        return None, None


with st.sidebar:
    st.title("⚙️ Configuration")

    st.subheader("API Keys")
    gemini_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        value=st.session_state.gemini_key,
        key="gemini_input",
    )
    pinecone_key = st.text_input(
        "Pinecone API Key",
        type="password",
        value=st.session_state.pinecone_key,
        key="pinecone_input",
    )

    if gemini_key:
        st.session_state.gemini_key = gemini_key
    if pinecone_key:
        st.session_state.pinecone_key = pinecone_key

    st.divider()

    st.subheader("📚 Load Documents")
    wiki_query = st.text_input(
        "Wikipedia Search Query", placeholder="e.g., Artificial Intelligence"
    )
    num_docs = st.slider("Number of articles", 1, 10, 5)

    if st.button("🔍 Load Documents", use_container_width=True):
        if not st.session_state.gemini_key or not st.session_state.pinecone_key:
            st.error("⚠️ Please provide both API keys!")
        elif not wiki_query:
            st.error("⚠️ Please enter a Wikipedia search query!")
        else:
            with st.spinner("Loading Wikipedia articles..."):
                try:
                    docs = WikipediaLoader(
                        query=wiki_query, load_max_docs=num_docs
                    ).load()
                    st.success(f"✅ Loaded {len(docs)} documents")

                    with st.expander("📄 Loaded Articles"):
                        for i, doc in enumerate(docs, 1):
                            st.write(f"{i}. {doc.metadata.get('title', 'Unknown')}")

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        add_start_index=True,
                    )
                    all_splits = text_splitter.split_documents(docs)
                    st.info(f"📝 Created {len(all_splits)} chunks")

                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/text-embedding-004",
                        google_api_key=st.session_state.gemini_key,
                    )

                    pc = Pinecone(api_key=st.session_state.pinecone_key)
                    index_name = "wikirag"

                    if index_name not in pc.list_indexes().names():
                        st.error(
                            f"❌ Index '{index_name}' not found. Please create it first with dimension 768."
                        )
                    else:
                        index = pc.Index(index_name)
                        vector_store = PineconeVectorStore(
                            embedding=embeddings, index=index
                        )

                        with st.spinner("Embedding and storing documents..."):
                            vector_store.add_documents(documents=all_splits)

                        st.success("✅ Documents embedded and stored!")

                        st.session_state.documents_loaded = True
                        st.success("🎉 System is ready! Start chatting below.")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.documents_loaded = False

    st.divider()

    if st.session_state.documents_loaded:
        st.success("✅ System Ready")
    else:
        st.warning("⚠️ Load documents to start chatting")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("🤖 Wikipedia RAG Chatbot")
st.markdown("Ask questions about the loaded Wikipedia articles!")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(
    "Ask me anything about the loaded documents...",
    disabled=not st.session_state.documents_loaded,
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if not st.session_state.gemini_key or not st.session_state.pinecone_key:
                    error_msg = (
                        "⚠️ API keys are missing. Please enter them in the sidebar."
                    )
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )
                else:
                    graph, vector_store = initialize_rag_system(
                        st.session_state.gemini_key, st.session_state.pinecone_key
                    )

                    if graph is None:
                        error_msg = (
                            "⚠️ Failed to initialize system. Check your API keys."
                        )
                        st.error(error_msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg}
                        )
                    else:
                        response = graph.invoke({"question": prompt})
                        answer = response["answer"]
                        st.markdown(answer)

                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer}
                        )

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )

if not st.session_state.documents_loaded and len(st.session_state.messages) == 0:
    st.info("👈 Please load documents from the sidebar to start chatting!")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Powered by LangChain, Google Gemini, and Pinecone
    </div>
    """,
    unsafe_allow_html=True,
)
