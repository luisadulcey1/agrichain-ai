
import json
import os
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# AgriChain AI - RAG Knowledge Assistant
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="AgriChain AI - RAG Knowledge Assistant",
    page_icon="🌱",
    layout="wide",
)

st.title("🌱 AgriChain AI - RAG Knowledge Assistant")
st.write(
    "Ask questions about AgriChain policies and receive "
    "AI-generated answers grounded in the knowledge base."
)


@st.cache_resource
def load_resources():
    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    index = faiss.read_index(
        str(MODELS_DIR / "knowledge_base.index")
    )

    with open(
        MODELS_DIR / "knowledge_base_metadata.json",
        "r",
        encoding="utf-8",
    ) as file:
        documents = json.load(file)

    if index.ntotal != len(documents):
        raise ValueError(
            "FAISS index and knowledge base metadata "
            "have different document counts."
        )

    return embedding_model, index, documents


try:
    embedding_model, index, documents = load_resources()
except Exception as error:
    st.error(f"Unable to load the knowledge base: {error}")
    st.stop()


if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []


for message in st.session_state.rag_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input(
    "Ask a question about delivery, refunds, or complaints..."
)

if question:
    st.session_state.rag_messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching AgriChain policies..."):
            try:
                embedding = embedding_model.encode(
                    [question],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ).astype("float32")

                scores, indices = index.search(
                    embedding,
                    min(3, index.ntotal),
                )

                retrieved_docs = []

                for idx in indices[0]:
                    if idx >= 0:
                        retrieved_docs.append(
                            documents[int(idx)]
                        )

                if not retrieved_docs:
                    answer = (
                        "I couldn't find relevant policy documents "
                        "in the knowledge base."
                    )
                else:
                    policy_context = "\n\n".join(
                        f"Document ID: {doc.get('doc_id', 'Unknown')}\n"
                        f"Policy: {doc.get('content', '')}"
                        for doc in retrieved_docs
                    )

                    llm = ChatOpenAI(
                        model=os.getenv(
                            "OPENAI_MODEL", "gpt-4.1-mini"
                        ),
                        temperature=0,
                    )

                    system_prompt = """
You are AgriChain AI's RAG Knowledge Assistant.

Answer the user's question using only the retrieved
AgriChain policy documents.

Rules:
1. Do not invent policies, compensation, or refunds.
2. Cite relevant document IDs such as [KB-00001].
3. If the documents do not contain the answer,
   clearly explain that the information is unavailable.
4. Do not claim that an investigation or refund
   has already been completed.
5. Respond in clear, professional English.
"""

                    response = llm.invoke([
                        SystemMessage(content=system_prompt),
                        HumanMessage(
                            content=(
                                f"USER QUESTION:\n{question}\n\n"
                                f"RETRIEVED POLICIES:\n"
                                f"{policy_context}"
                            )
                        ),
                    ])

                    answer = str(response.content)

                st.markdown(answer)

                if retrieved_docs:
                    with st.expander("Retrieved Policy Sources"):
                        for doc in retrieved_docs:
                            st.markdown(
                                f"**{doc.get('doc_id', 'Unknown')}**"
                            )
                            st.write(doc.get("content", ""))

                st.session_state.rag_messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as error:
                st.error(
                    f"Unable to generate an answer: {error}"
                )
