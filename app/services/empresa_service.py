# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
import glob as glob_module  # Renombramos para evitar confusión con el parámetro glob
from langchain_ollama import OllamaLLM
from langchain.memory import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

import os
import torch


class EmpresaService:
    def __init__(self):
        # Use Apple Silicon acceleration if available
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        # self.model_name = "tiiuae/falcon-rw-1b"
        # self.model_name = "tinyllama:latest"
        self.model_name = "gemma3:latest"

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.persist_dir = "chroma_db"

        # Load or create vector store
        if os.path.exists(self.persist_dir):
            self.vector_store = Chroma(
                embedding_function=self.embeddings, persist_directory=self.persist_dir
            )
        else:
            documents = self.load_documents()
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            split_docs = text_splitter.split_documents(documents)

            if not split_docs:
                raise ValueError("Error: split_docs is empty")

            self.vector_store = Chroma.from_documents(
                split_docs, self.embeddings, persist_directory=self.persist_dir
            )

        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Usa SOLO la siguiente información para responder la pregunta sobre empresas. "
                "NO inventes datos. Responde específicamente sobre la empresa mencionada en la pregunta. "
                "Si no encuentras información sobre la empresa específica, di 'No tengo información sobre esa empresa'.\n\n"
                "Información:\n{context}\n\n"
                "Pregunta: {question}\nRespuesta:"
            ),
        )

        # Create base chain without memory
        base_qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.load_llm(),
            retriever=self.vector_store.as_retriever(),
            return_source_documents=False,
            combine_docs_chain_kwargs={"prompt": self.prompt},
            verbose=True,
        )
        # Store chat histories in a dictionary
        self.chat_histories = {}

        def get_session_history(session_id: str) -> ChatMessageHistory:
            if session_id not in self.chat_histories:
                self.chat_histories[session_id] = ChatMessageHistory()
            return self.chat_histories[session_id]

        # Wrap with RunnableWithMessageHistory
        self.qa_chain = RunnableWithMessageHistory(
            base_qa_chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

    def ask(self, query: str, session_id: str = "default") -> str:
        response = self.qa_chain.invoke(
            {"question": query}, config={"configurable": {"session_id": session_id}}
        )
        return response.get("answer", "No se pudo generar una respuesta.")

    def add_metadata(self, doc, doc_type):
        doc.metadata["doc_type"] = doc_type
        return doc

    def load_documents(self):
        folder = os.path.join(os.path.dirname(__file__), "..", "repository")
        folder = os.path.abspath(folder)

        print(f"Buscando documentos en: {folder}")

        if not os.path.exists(folder):
            print(f"ERROR: El directorio '{folder}' no existe")
            raise ValueError(f"Directory '{folder}' does not exist!")

        documents = []

        # Usar glob directamente para encontrar archivos
        patterns = ["*.md", "*.txt", "**/*.md", "**/*.txt"]
        for pattern in patterns:
            # Construye la ruta completa del patrón
            full_pattern = os.path.join(folder, pattern)
            print(f"Buscando con patrón: {full_pattern}")

            # Usa glob.glob() para búsquedas básicas o glob.iglob() para búsquedas recursivas
            if "**" in pattern:
                # glob.glob() con recursive=True para patrones con **
                matching_files = glob_module.glob(full_pattern, recursive=True)
            else:
                # glob.glob() normal para patrones simples
                matching_files = glob_module.glob(full_pattern)

            print(f"  Encontrados {len(matching_files)} archivos")

            # Cargar cada archivo encontrado
            for file_path in matching_files:
                print(f"  Cargando: {file_path}")
                try:
                    loader = TextLoader(file_path, encoding="utf-8")
                    file_docs = loader.load()
                    documents.extend(file_docs)
                    print(f"    Cargado exitosamente: {len(file_docs)} documentos")
                except Exception as e:
                    print(f"    Error al cargar {file_path}: {str(e)}")

        print(f"Total de documentos cargados: {len(documents)}")

        if not documents:
            raise ValueError("No documents loaded from repository!")

        return documents

    def load_llm(self):
        print("Loading Ollama model")
        return OllamaLLM(
            model=self.model_name,
            temperature=0.7,  # Adjust for more/less creative responses
            top_p=0.9,
            # timeout=60  # Optional timeout
        )
