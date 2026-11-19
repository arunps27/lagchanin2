# ============================================================
# GOOGLE COLAB - SINGLE CELL
#
# LANGCHAIN + LANGGRAPH + GROQ + GRADIO
# DYNAMIC RAG + DYNAMIC DOCUMENT GRAPH
#
# LLM:
#     openai/gpt-oss-120b
#
# Features:
#   1. Groq API key entered in UI
#   2. Upload PDF / DOCX / TXT / MD / CSV
#   3. LangChain document loading
#   4. Chunking
#   5. HuggingFace embeddings
#   6. FAISS vector store
#   7. LangGraph RAG workflow
#   8. Dynamic document knowledge graph
#   9. Chatbot using Gradio messages format
#  10. Graph updates after every upload/build
# ============================================================


# ============================================================
# 1. INSTALL ALL PACKAGES
# ============================================================

# (install deps with: pip install -r requirements, see README/comments)


# ============================================================
# 2. IMPORTS
# ============================================================

import os
import traceback
from pathlib import Path
from typing import TypedDict, Any

import gradio as gr
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go

from langchain_groq import ChatGroq

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader
)

from langchain_community.vectorstores import FAISS

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import (
    StateGraph,
    START,
    END
)


# ============================================================
# 3. GLOBAL APPLICATION STATE
# ============================================================

VECTORSTORE = None

ALL_CHUNKS = []

UPLOADED_FILES = []

DOCUMENT_COUNT = 0

CHUNK_COUNT = 0

GRAPH = None

MAX_RETRIEVE_K = 50   # upper bound on chunks sent to the LLM

RETRIEVE_K = 4


# ============================================================
# 4. EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(

    model_name=
    "sentence-transformers/all-MiniLM-L6-v2",

    model_kwargs={
        "device": "cpu"
    },

    encode_kwargs={
        "normalize_embeddings": True
    }
)

print("Embedding model loaded.")


# ============================================================
# 5. LOAD SINGLE FILE
# ============================================================

def load_single_file(file_path):

    suffix = Path(
        file_path
    ).suffix.lower()

    if suffix == ".pdf":

        loader = PyPDFLoader(
            file_path
        )

    elif suffix in [".txt", ".md"]:

        loader = TextLoader(
            file_path,
            encoding="utf-8",
            autodetect_encoding=True
        )

    elif suffix == ".csv":

        loader = CSVLoader(
            file_path
        )

    elif suffix == ".docx":

        loader = Docx2txtLoader(
            file_path
        )

    else:

        raise ValueError(
            f"Unsupported file format: {suffix}"
        )

    documents = loader.load()

    for doc in documents:

        doc.metadata[
            "source_file"
        ] = Path(
            file_path
        ).name

    return documents


# ============================================================
# 6. LOAD ALL FILES
# ============================================================

def load_all_files(files):

    if not files:

        raise ValueError(
            "Please upload at least one file."
        )

    documents = []

    for file_path in files:

        docs = load_single_file(
            file_path
        )

        documents.extend(
            docs
        )

    return documents


# ============================================================
# 7. BUILD KNOWLEDGE BASE
# ============================================================

def build_knowledge_base(files):

    global VECTORSTORE
    global ALL_CHUNKS
    global UPLOADED_FILES
    global DOCUMENT_COUNT
    global CHUNK_COUNT
    global RETRIEVE_K

    try:

        # ----------------------------------------
        # Reset
        # ----------------------------------------

        VECTORSTORE = None

        ALL_CHUNKS = []

        UPLOADED_FILES = []

        # ----------------------------------------
        # Load
        # ----------------------------------------

        documents = load_all_files(
            files
        )

        DOCUMENT_COUNT = len(
            documents
        )

        UPLOADED_FILES = [
            Path(f).name
            for f in files
        ]

        # ----------------------------------------
        # Split
        # ----------------------------------------

        splitter = RecursiveCharacterTextSplitter(

            chunk_size=800,

            chunk_overlap=100,

            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

        chunks = splitter.split_documents(
            documents
        )

        if not chunks:

            raise ValueError(
                "No text could be extracted."
            )

        # ----------------------------------------
        # Add chunk IDs
        # ----------------------------------------

        for index, chunk in enumerate(
            chunks
        ):

            chunk.metadata[
                "chunk_id"
            ] = index + 1

        ALL_CHUNKS = chunks

        RETRIEVE_K = min(len(chunks), MAX_RETRIEVE_K)

        CHUNK_COUNT = len(
            chunks
        )

        # ----------------------------------------
        # FAISS
        # ----------------------------------------

        VECTORSTORE = FAISS.from_documents(

            chunks,

            embeddings
        )

        # ----------------------------------------
        # Dynamic graph
        # ----------------------------------------

        graph_image = create_sphere_graph(
            documents,
            chunks
        )

        # ----------------------------------------
        # Status
        # ----------------------------------------

        status = (
            "## ✅ Knowledge Base Created\n\n"
            f"**Files:** {len(files)}  \n"
            f"**Documents/Pages:** {DOCUMENT_COUNT}  \n"
            f"**Chunks:** {CHUNK_COUNT}\n\n"
            "### Uploaded Files\n\n"
        )

        for filename in UPLOADED_FILES:

            status += (
                f"- 📄 `{filename}`\n"
            )

        status += (
            "\n### RAG Configuration\n\n"
            "- Embeddings: `all-MiniLM-L6-v2`\n"
            "- Vector DB: `FAISS`\n"
            f"- Retrieval: top {RETRIEVE_K} chunks (maximum, capped at {MAX_RETRIEVE_K})\n"
            "- LLM: `openai/gpt-oss-120b`"
        )

        return (
            status,
            graph_image
        )

    except Exception as e:

        print(
            traceback.format_exc()
        )

        VECTORSTORE = None

        return (
            f"## ❌ Build Error\n\n`{str(e)}`",
            None
        )


# ============================================================
# 8. CREATE DYNAMIC DOCUMENT GRAPH
# ============================================================
#
# This graph is generated from the ACTUAL uploaded data.
#
# Structure:
#
# File
#   |
#   +--> Chunk 1
#   +--> Chunk 2
#   +--> Chunk 3
#
# For multiple files:
#
# File A ---> chunks
# File B ---> chunks
# File C ---> chunks
#
# This means every upload produces a NEW graph.
# ============================================================

def create_document_graph(
    documents,
    chunks
):

    try:

        G = nx.DiGraph()

        # ----------------------------------------
        # Add source nodes
        # ----------------------------------------

        source_files = []

        for doc in documents:

            source = doc.metadata.get(
                "source_file",
                "Unknown"
            )

            if source not in source_files:

                source_files.append(
                    source
                )

        # ----------------------------------------
        # File nodes
        # ----------------------------------------

        for source in source_files:

            G.add_node(

                f"FILE::{source}",

                node_type="file",

                label=source
            )

        # ----------------------------------------
        # Chunk nodes
        # ----------------------------------------

        for chunk in chunks:

            chunk_id = chunk.metadata.get(
                "chunk_id"
            )

            source = chunk.metadata.get(
                "source_file",
                "Unknown"
            )

            chunk_node = (
                f"CHUNK::{chunk_id}"
            )

            preview = (
                chunk.page_content
                .replace("\n", " ")
                .strip()
            )

            preview = preview[:100]

            G.add_node(

                chunk_node,

                node_type="chunk",

                label=(
                    f"Chunk {chunk_id}\n"
                    f"{preview}"
                )
            )

            # File -> Chunk

            G.add_edge(

                f"FILE::{source}",

                chunk_node
            )

        # ----------------------------------------
        # Add RAG nodes
        # ----------------------------------------

        G.add_node(

            "VECTOR",

            node_type="vector",

            label="FAISS\nVector Store"
        )

        G.add_node(

            "RETRIEVER",

            node_type="retriever",

            label="Retriever\nTop-K"
        )

        G.add_node(

            "LLM",

            node_type="llm",

            label="Groq\nGPT-OSS-120B"
        )

        # ----------------------------------------
        # Connect chunks to vector store
        # ----------------------------------------

        for chunk in chunks:

            chunk_id = chunk.metadata.get(
                "chunk_id"
            )

            G.add_edge(

                f"CHUNK::{chunk_id}",

                "VECTOR"
            )

        # ----------------------------------------
        # RAG flow
        # ----------------------------------------

        G.add_edge(
            "VECTOR",
            "RETRIEVER"
        )

        G.add_edge(
            "RETRIEVER",
            "LLM"
        )

        # ----------------------------------------
        # Draw
        # ----------------------------------------

        fig, ax = plt.subplots(
            figsize=(16, 10)
        )

        # Layered positions
        pos = {}

        # ----------------------------------------
        # File positions
        # ----------------------------------------

        file_nodes = [
            n for n, d in G.nodes(
                data=True
            )
            if d["node_type"] == "file"
        ]

        for i, node in enumerate(
            file_nodes
        ):

            pos[node] = (
                0,
                -i * 2
            )

        # ----------------------------------------
        # Chunk positions
        # ----------------------------------------

        chunk_nodes = [
            n for n, d in G.nodes(
                data=True
            )
            if d["node_type"] == "chunk"
        ]

        chunks_by_file = {}

        for node in chunk_nodes:

            source = None

            for parent in G.predecessors(
                node
            ):

                if G.nodes[parent][
                    "node_type"
                ] == "file":

                    source = parent

                    break

            if source is not None:

                chunks_by_file.setdefault(
                    source,
                    []
                ).append(node)

        for file_index, (
            source,
            node_list
        ) in enumerate(
            chunks_by_file.items()
        ):

            for chunk_index, node in enumerate(
                node_list
            ):

                pos[node] = (
                    3,
                    -(file_index * 3)
                    -
                    chunk_index * 0.7
                )

        # ----------------------------------------
        # RAG nodes
        # ----------------------------------------

        pos["VECTOR"] = (
            6,
            -2
        )

        pos["RETRIEVER"] = (
            8.5,
            -2
        )

        pos["LLM"] = (
            11,
            -2
        )

        # ----------------------------------------
        # Node grouping
        # ----------------------------------------

        file_group = [
            n for n, d in G.nodes(
                data=True
            )
            if d["node_type"] == "file"
        ]

        chunk_group = [
            n for n, d in G.nodes(
                data=True
            )
            if d["node_type"] == "chunk"
        ]

        vector_group = [
            n for n, d in G.nodes(
                data=True
            )
            if d["node_type"] == "vector"
        ]

        retriever_group = [
            n for n, d in G.nodes(
                data=True
            )
            if d["node_type"] == "retriever"
        ]

        llm_group = [
            n for n, d in G.nodes(
                data=True
            )
            if d["node_type"] == "llm"
        ]

        # ----------------------------------------
        # Draw files
        # ----------------------------------------

        nx.draw_networkx_nodes(

            G,
            pos,

            nodelist=file_group,

            node_size=5000,

            node_color="lightblue",

            edgecolors="black",

            linewidths=1.5,

            ax=ax
        )

        # ----------------------------------------
        # Draw chunks
        # ----------------------------------------

        nx.draw_networkx_nodes(

            G,
            pos,

            nodelist=chunk_group,

            node_size=3000,

            node_color="lightgreen",

            edgecolors="black",

            ax=ax
        )

        # ----------------------------------------
        # Vector
        # ----------------------------------------

        nx.draw_networkx_nodes(

            G,
            pos,

            nodelist=vector_group,

            node_size=5000,

            node_color="plum",

            edgecolors="black",

            ax=ax
        )

        # ----------------------------------------
        # Retriever
        # ----------------------------------------

        nx.draw_networkx_nodes(

            G,
            pos,

            nodelist=retriever_group,

            node_size=5000,

            node_color="khaki",

            edgecolors="black",

            ax=ax
        )

        # ----------------------------------------
        # LLM
        # ----------------------------------------

        nx.draw_networkx_nodes(

            G,
            pos,

            nodelist=llm_group,

            node_size=5000,

            node_color="salmon",

            edgecolors="black",

            ax=ax
        )

        # ----------------------------------------
        # Edges
        # ----------------------------------------

        nx.draw_networkx_edges(

            G,

            pos,

            arrows=True,

            arrowsize=20,

            width=1.5,

            alpha=0.7,

            ax=ax
        )

        # ----------------------------------------
        # Labels
        # ----------------------------------------

        labels = {

            node:
            data["label"]

            for node, data

            in G.nodes(
                data=True
            )
        }

        nx.draw_networkx_labels(

            G,

            pos,

            labels=labels,

            font_size=8,

            font_weight="bold",

            ax=ax
        )

        ax.set_title(

            (
                f"Dynamic Document + RAG Graph\n"
                f"{len(source_files)} files | "
                f"{len(chunks)} chunks"
            ),

            fontsize=18,

            fontweight="bold"
        )

        ax.axis("off")

        plt.tight_layout()

        output_file = (
            "dynamic_rag_graph.png"
        )

        plt.savefig(

            output_file,

            dpi=150,

            bbox_inches="tight"
        )

        plt.close(fig)

        return output_file

    except Exception as e:

        print(
            "Graph error:",
            traceback.format_exc()
        )

        return None



def _slerp(a, b, n=14):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    omega = np.arccos(np.clip(np.dot(a, b), -1, 1))
    if omega < 1e-6:
        return np.array([a, b])
    t = np.linspace(0, 1, n)[:, None]
    return (np.sin((1 - t) * omega) * a + np.sin(t * omega) * b) / np.sin(omega)


def create_sphere_graph(documents, chunks):
    """Interactive 3D graph with every node placed on a sphere."""
    try:
        files = []
        for c in chunks:
            f = c.metadata.get("source_file", "Unknown")
            if f not in files:
                files.append(f)

        nodes = []
        for f in files:
            nodes.append((f"F::{f}", "file", f, f"<b>File</b><br>{f}"))
        for c in chunks:
            cid = c.metadata.get("chunk_id")
            prev = c.page_content.replace("\n", " ").strip()[:200]
            nodes.append((f"C::{cid}", "chunk", f"C{cid}",
                          f"<b>Chunk {cid}</b> ({c.metadata.get('source_file')})<br>{prev}"))
        nodes.append(("VECTOR", "vector", "FAISS", "<b>FAISS Vector Store</b>"))
        nodes.append(("RETRIEVER", "retriever", "Retriever",
                      f"<b>Retriever</b><br>top-{RETRIEVE_K} chunks"))
        nodes.append(("LLM", "llm", "LLM", "<b>Groq GPT-OSS-120B</b>"))

        n = len(nodes)
        i = np.arange(n) + 0.5
        phi = np.arccos(1 - 2 * i / n)
        theta = np.pi * (1 + 5 ** 0.5) * i
        xyz = np.stack([np.cos(theta) * np.sin(phi),
                        np.sin(theta) * np.sin(phi),
                        np.cos(phi)], axis=1)
        idx = {nd[0]: k for k, nd in enumerate(nodes)}

        edges = []
        for c in chunks:
            cid = c.metadata.get("chunk_id")
            f = c.metadata.get("source_file", "Unknown")
            edges.append((f"F::{f}", f"C::{cid}", "file->chunk"))
            edges.append((f"C::{cid}", "VECTOR", "chunk->vector"))
        edges += [("VECTOR", "RETRIEVER", "flow"), ("RETRIEVER", "LLM", "flow")]

        colors = {"file": "dodgerblue", "chunk": "limegreen", "vector": "purple",
                  "retriever": "gold", "llm": "tomato"}
        sizes = {"file": 14, "chunk": 6, "vector": 14, "retriever": 14, "llm": 14}

        fig = go.Figure()

        for name, col, wid in [("file->chunk", "rgba(30,144,255,0.45)", 2),
                               ("chunk->vector", "rgba(150,150,150,0.18)", 1),
                               ("flow", "rgba(220,60,60,0.9)", 5)]:
            ex, ey, ez = [], [], []
            for a, b, kind in edges:
                if kind != name:
                    continue
                arc = _slerp(xyz[idx[a]], xyz[idx[b]])
                ex += list(arc[:, 0]) + [None]
                ey += list(arc[:, 1]) + [None]
                ez += list(arc[:, 2]) + [None]
            fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                                       line=dict(color=col, width=wid),
                                       hoverinfo="skip", name=name))

        for kind in colors:
            sel = [k for k, nd in enumerate(nodes) if nd[1] == kind]
            if not sel:
                continue
            fig.add_trace(go.Scatter3d(
                x=xyz[sel, 0], y=xyz[sel, 1], z=xyz[sel, 2],
                mode="markers+text" if kind != "chunk" else "markers",
                text=[nodes[k][2] for k in sel],
                textposition="top center",
                hovertext=[nodes[k][3] for k in sel], hoverinfo="text",
                marker=dict(size=sizes[kind], color=colors[kind],
                            line=dict(color="black", width=1)),
                name=kind))

        u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
        fig.add_trace(go.Surface(
            x=0.98 * np.cos(u) * np.sin(v), y=0.98 * np.sin(u) * np.sin(v),
            z=0.98 * np.cos(v), opacity=0.06, showscale=False,
            colorscale=[[0, "lightblue"], [1, "lightblue"]],
            hoverinfo="skip", showlegend=False))

        axis = dict(visible=False)
        fig.update_layout(
            title=f"Spherical Document + RAG Graph - {len(files)} files, "
                  f"{len(chunks)} chunks (drag to rotate, scroll to zoom)",
            scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="cube"),
            margin=dict(l=0, r=0, t=50, b=0), height=700,
            legend=dict(itemsizing="constant"))
        return fig

    except Exception:
        print("Sphere graph error:", traceback.format_exc())
        return None


# ============================================================
# 9. LANGGRAPH STATE
# ============================================================

class RAGState(TypedDict):

    question: str

    context: str

    answer: str

    llm: Any


# ============================================================
# 10. LANGGRAPH RETRIEVAL NODE
# ============================================================

def retrieve_documents(
    state: RAGState
):

    question = state[
        "question"
    ]

    if VECTORSTORE is None:

        return {

            "context":
            "No documents have been uploaded."
        }

    docs = VECTORSTORE.similarity_search(

        question,

        k=RETRIEVE_K
    )

    context_parts = []

    for index, doc in enumerate(
        docs,
        start=1
    ):

        source = doc.metadata.get(

            "source_file",

            "Unknown"
        )

        context_parts.append(

            f"[Source {index}: {source}]\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(
        context_parts
    )

    return {

        "context":
        context
    }


# ============================================================
# 11. LANGGRAPH GENERATION NODE
# ============================================================

def generate_answer(
    state: RAGState
):

    question = state[
        "question"
    ]

    context = state[
        "context"
    ]

    llm = state[
        "llm"
    ]

    prompt = ChatPromptTemplate.from_messages(

        [

            (
                "system",

                """
You are a RAG document assistant.

Use ONLY the provided document context
to answer the question.

Rules:

- Do not invent facts.
- If the answer is not present,
  clearly say so.
- Be concise and accurate.
- Mention the source filename when helpful.

DOCUMENT CONTEXT:

{context}
"""
            ),

            (
                "human",

                "{question}"
            )
        ]
    )

    chain = prompt | llm

    response = chain.invoke(

        {

            "context":
            context,

            "question":
            question
        }
    )

    return {

        "answer":
        response.content
    }


# ============================================================
# 12. BUILD LANGGRAPH
# ============================================================

workflow = StateGraph(
    RAGState
)

workflow.add_node(
    "retrieve_documents",
    retrieve_documents
)

workflow.add_node(
    "generate_answer",
    generate_answer
)

workflow.add_edge(
    START,
    "retrieve_documents"
)

workflow.add_edge(
    "retrieve_documents",
    "generate_answer"
)

workflow.add_edge(
    "generate_answer",
    END
)

GRAPH = workflow.compile()


# ============================================================
# 13. LANGGRAPH WORKFLOW IMAGE
# ============================================================

def create_workflow_image():

    try:

        graph_data = GRAPH.get_graph()

        G = nx.DiGraph()

        for node in graph_data.nodes:

            name = str(node)

            if name == "__start__":

                label = "START"

            elif name == "__end__":

                label = "END"

            else:

                label = (
                    name
                    .replace(
                        "_",
                        "\n"
                    )
                    .title()
                )

            G.add_node(
                name,
                label=label
            )

        for edge in graph_data.edges:

            G.add_edge(

                str(edge.source),

                str(edge.target)
            )

        fig, ax = plt.subplots(

            figsize=(12, 5)
        )

        pos = nx.spring_layout(

            G,

            seed=42,

            k=2
        )

        nx.draw_networkx_nodes(

            G,

            pos,

            node_size=5000,

            node_color="lightblue",

            edgecolors="black",

            ax=ax
        )

        nx.draw_networkx_edges(

            G,

            pos,

            arrows=True,

            arrowsize=25,

            width=2,

            ax=ax
        )

        labels = {

            node: data["label"]

            for node, data in
            G.nodes(
                data=True
            )
        }

        nx.draw_networkx_labels(

            G,

            pos,

            labels=labels,

            font_size=11,

            font_weight="bold",

            ax=ax
        )

        ax.set_title(

            "LangGraph RAG Execution Workflow",

            fontsize=17,

            fontweight="bold"
        )

        ax.axis("off")

        plt.tight_layout()

        output_file = (
            "langgraph_workflow.png"
        )

        plt.savefig(

            output_file,

            dpi=160,

            bbox_inches="tight"
        )

        plt.close(fig)

        return output_file

    except Exception as e:

        print(
            "Workflow graph error:",
            e
        )

        return None


WORKFLOW_IMAGE = (
    create_workflow_image()
)


# ============================================================
# 14. CREATE GROQ LLM
# ============================================================

def create_llm(
    groq_api_key
):

    if not groq_api_key:

        raise ValueError(
            "Please enter your Groq API key."
        )

    groq_api_key = (
        groq_api_key.strip()
    )

    if not groq_api_key:

        raise ValueError(
            "Please enter your Groq API key."
        )

    return ChatGroq(

        api_key=groq_api_key,

        model=
        "openai/gpt-oss-120b",

        temperature=0,

        max_tokens=2000,

        max_retries=2
    )


# ============================================================
# 15. CHAT FUNCTION
# ============================================================
#
# IMPORTANT:
#
# Your Gradio error:
#
# Data incompatible with messages format
#
# is fixed here by ALWAYS returning:
#
# [
#   {"role": "user", "content": "..."},
#   {"role": "assistant", "content": "..."}
# ]
#
# No tuple pairs are used.
# ============================================================

def chat_with_rag(

    message,

    history,

    groq_api_key
):

    try:

        history = history or []

        if not message or not message.strip():

            return history

        message = message.strip()

        # ----------------------------------------
        # Create LLM
        # ----------------------------------------

        llm = create_llm(
            groq_api_key
        )

        # ----------------------------------------
        # Knowledge base check
        # ----------------------------------------

        if VECTORSTORE is None:

            answer = (

                "Please upload your documents "
                "and click **Build Knowledge Base** "
                "before asking a question."
            )

        else:

            # ------------------------------------
            # Execute LangGraph
            # ------------------------------------

            result = GRAPH.invoke(

                {

                    "question":
                    message,

                    "context":
                    "",

                    "answer":
                    "",

                    "llm":
                    llm
                }
            )

            answer = result.get(

                "answer",

                "No answer generated."
            )

        # ----------------------------------------
        # IMPORTANT:
        # Messages format
        # ----------------------------------------

        history.append(

            {

                "role":
                "user",

                "content":
                message
            }
        )

        history.append(

            {

                "role":
                "assistant",

                "content":
                answer
            }
        )

        return history

    except Exception as e:

        print(
            traceback.format_exc()
        )

        history = history or []

        history.append(

            {

                "role":
                "user",

                "content":
                message
            }
        )

        history.append(

            {

                "role":
                "assistant",

                "content":
                f"❌ Error: {str(e)}"
            }
        )

        return history


# ============================================================
# 16. CLEAR CHAT
# ============================================================

def clear_chat():

    return []


# ============================================================
# 17. CSS
# ============================================================

custom_css = """

#app_title {
    text-align: center;
    font-size: 30px;
    font-weight: 700;
}

#app_subtitle {
    text-align: center;
    font-size: 15px;
    opacity: 0.8;
}

"""


# ============================================================
# 18. GRADIO UI
# ============================================================

with gr.Blocks(

    title=
    "LangChain + LangGraph Dynamic RAG"

) as demo:

    # ----------------------------------------
    # Header
    # ----------------------------------------

    gr.Markdown(

        """
# 🧠 LangChain + LangGraph Dynamic RAG

Upload documents → Build Knowledge Base →
Generate Dynamic Data Graph → Ask Questions
""",

        elem_id=
        "app_title"
    )

    gr.Markdown(

        """
**LLM:** `openai/gpt-oss-120b` via Groq
**Vector Store:** FAISS
**Embeddings:** all-MiniLM-L6-v2
""",

        elem_id=
        "app_subtitle"
    )


    # ========================================================
    # GROQ
    # ========================================================

    gr.Markdown(
        "## 🔐 Groq API Configuration"
    )

    with gr.Row():

        groq_api_key = gr.Textbox(

            label=
            "Groq API Key",

            placeholder=
            "Enter your Groq API key",

            type="password",

            scale=3
        )

        gr.Markdown(

            """
### Model

`openai/gpt-oss-120b`

The API key is supplied by the
user through the UI.
""",

            scale=1
        )


    # ========================================================
    # FILES
    # ========================================================

    gr.Markdown(
        "## 📚 Upload Documents"
    )

    with gr.Row():

        with gr.Column(
            scale=1
        ):

            files = gr.File(

                label=
                "PDF / DOCX / TXT / MD / CSV",

                file_count="multiple",

                file_types=[

                    ".pdf",

                    ".docx",

                    ".txt",

                    ".md",

                    ".csv"
                ],

                type="filepath"
            )

            build_button = gr.Button(

                "🚀 Build Knowledge Base",

                variant="primary"
            )


        with gr.Column(
            scale=1
        ):

            status = gr.Markdown(

                """
### 📂 Status

No documents loaded yet.
"""
            )


    # ========================================================
    # DYNAMIC DATA GRAPH
    # ========================================================

    gr.Markdown(
        "## 📊 Dynamic Document + RAG Graph"
    )

    gr.Markdown(

        """
This graph is regenerated from the actual uploaded data.

**File → Chunks → FAISS → Retriever → GPT-OSS-120B**
"""
    )

    document_graph = gr.Plot(label="Spherical Data Graph")


    # ========================================================
    # LANGGRAPH WORKFLOW
    # ========================================================

    gr.Markdown(
        "## 🔗 LangGraph Workflow"
    )

    workflow_graph = gr.Image(

        value=
        WORKFLOW_IMAGE,

        label=
        "LangGraph Nodes & Edges",

        type="filepath"
    )


    gr.Markdown(

        """
### LangGraph Execution

`START`
→
`retrieve_documents`
→
`generate_answer`
→
`END`
"""
    )


    # ========================================================
    # CHAT
    # ========================================================

    gr.Markdown(
        "## 💬 Chat with Your Documents"
    )

    chatbot = gr.Chatbot(

        label=
        "RAG Assistant",

        height=500,

        placeholder=
        "Ask a question about your uploaded documents..."
    )


    with gr.Row():

        message = gr.Textbox(

            label=
            "Your Question",

            placeholder=
            "Ask something about the uploaded documents...",

            scale=8
        )

        send_button = gr.Button(

            "Send",

            variant="primary",

            scale=1
        )

        clear_button = gr.Button(

            "Clear",

            scale=1
        )


    # ========================================================
    # EVENTS
    # ========================================================

    build_button.click(

        fn=
        build_knowledge_base,

        inputs=[
            files
        ],

        outputs=[

            status,

            document_graph
        ]
    )


    send_button.click(

        fn=
        chat_with_rag,

        inputs=[

            message,

            chatbot,

            groq_api_key
        ],

        outputs=[

            chatbot
        ]
    ).then(

        fn=
        lambda: "",

        outputs=[
            message
        ]
    )


    message.submit(

        fn=
        chat_with_rag,

        inputs=[

            message,

            chatbot,

            groq_api_key
        ],

        outputs=[

            chatbot
        ]
    ).then(

        fn=
        lambda: "",

        outputs=[
            message
        ]
    )


    clear_button.click(

        fn=
        clear_chat,

        inputs=[],

        outputs=[
            chatbot
        ]
    )


# ============================================================
# 19. LAUNCH
# ============================================================
#
# Gradio 6:
# CSS belongs in launch(), not Blocks().
# ============================================================

demo.launch(

    share=False,

    debug=False,

    css=custom_css
)