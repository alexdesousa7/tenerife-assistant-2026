"""
rag.py – Pipeline RAG (Retrieval-Augmented Generation) para el Asistente de Tenerife
-----------------------------------------------------------------------------------

Este módulo implementa el pipeline completo de RAG:
1. Carga de PDF con PyPDF
2. Segmentación de texto en chunks (LangChain)
3. Generación de embeddings con OpenAI (o fallback local en CI/tests)
4. Indexación con FAISS
5. Búsqueda semántica de chunks relevantes (con metadatos para citación)
"""

import logging
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv

# Configuración de logging
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Configuración de clientes y modelos
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# Cliente OpenAI (singleton)
_client: Optional[OpenAI] = None


def _get_client() -> Optional[OpenAI]:
    """
    Obtiene o crea el cliente de OpenAI (singleton).

    En entorno sin OPENAI_API_KEY (por ejemplo, CI / tests),
    devuelve None para permitir un fallback sin red.
    """
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            logger.warning(
                "OPENAI_API_KEY no está definida. "
                "Se usará un modo de embeddings sin OpenAI (tests/CI)."
            )
            return None
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client

# 1. Carga de documentos en este caso PDF

def load_pdf(pdf_path: str) -> List[str]:
    """
    Extrae el texto del PDF y devuelve una lista de textos por página.
    Cada elemento de la lista corresponde a una página.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"El archivo PDF no existe: {pdf_path}")

    logger.info(f"Cargando PDF desde: {pdf_path}")

    try:
        reader = PdfReader(pdf_path)
        if not reader.pages:
            raise ValueError(f"El PDF está vacío o no tiene páginas: {pdf_path}")

        pages: List[str] = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            pages.append(page_text)
            logger.debug(f"Página {i+1}: {len(page_text)} caracteres")

        logger.info(f"PDF cargado: {len(pages)} páginas")
        return pages

    except Exception as e:
        logger.error(f"Error al leer PDF: {e}")
        raise

# 2. Segmentacion del texto en chunks con metadatos para citación

def chunk_text(
    pages: List[str],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> (List[str], List[Dict[str, Any]]):
    """
    Divide el texto de cada página en fragmentos solapados y genera metadatos
    por chunk (página de origen, descripción de fuente).
    """
    logger.info(f"Segmentando texto: chunk_size={chunk_size}, overlap={chunk_overlap}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )

    chunks: List[str] = []
    metadata: List[Dict[str, Any]] = []

    for page_num, page_text in enumerate(pages):
        page_chunks = splitter.split_text(page_text)
        for ch in page_chunks:
            if not ch.strip():
                continue
            chunks.append(ch)
            metadata.append(
                {
                    "page": page_num + 1,
                    "source": f"Guía turística (pág. {page_num + 1})",
                }
            )

    logger.info(f"Texto segmentado en {len(chunks)} chunks con metadatos")
    return chunks, metadata

# 3. Generacion de embeddings (OpenAI o fallback local para CI/tessts)

def embed_chunks(chunks: List[str]) -> np.ndarray:
    """
    Genera embeddings para una lista de chunks.

    - Si hay OPENAI_API_KEY → usa OpenAI Embeddings.
    - Si NO hay clave (CI/tests) → usa embeddings deterministas locales
      basados en hash del texto (suficiente para que los tests pasen).
    """
    client = _get_client()
    logger.info(f"Generando embeddings para {len(chunks)} chunks...")

    # 🔹 Modo sin OpenAI (CI / tests)
    if client is None:
        dim = 256
        vecs = []
        for text in chunks:
            seed = abs(hash(text)) % (2**32)
            rng = np.random.default_rng(seed)
            vec = rng.normal(size=dim).astype("float32")
            vecs.append(vec)
        embeddings = np.stack(vecs, axis=0)
        logger.info(f"Embeddings locales generados: forma {embeddings.shape}")
        return embeddings

    # 🔹 Modo normal con OpenAI
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=chunks
        )

        embeddings = np.array([item.embedding for item in response.data]).astype("float32")
        logger.info(f"Embeddings generados: forma {embeddings.shape}")
        return embeddings

    except Exception as e:
        logger.error(f"Error al generar embeddings: {e}")
        raise RuntimeError(f"Fallo en generación de embeddings: {e}")

# 4. Construcción de indice FAISS

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Construye un índice FAISS para búsquedas eficientes."""
    logger.info("Construyendo índice FAISS...")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    logger.info(f"Índice FAISS creado con {index.ntotal} vectores")
    return index

# 5. Busqueda Semantica con recuperación de metadatos para citación.

def retrieve(
    query: str,
    chunks: List[str],
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    k: int = 4
) -> List[Dict[str, Any]]:
    """
    Recupera los chunks más relevantes para una consulta, incluyendo metadatos
    (página y descripción de fuente) para citación posterior.
    """
    client = _get_client()
    logger.info(f"Buscando {k} chunks relevantes para: '{query[:50]}...'")

    try:
        # 🔹 Modo sin OpenAI (CI / tests): usamos un embedding local para la query
        if client is None:
            dim = index.d
            seed = abs(hash(query)) % (2**32)
            rng = np.random.default_rng(seed)
            query_embedding = rng.normal(size=dim).astype("float32").reshape(1, -1)
        else:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[query]
            )
            query_embedding = np.array(response.data[0].embedding).astype("float32").reshape(1, -1)

        distances, indices = index.search(query_embedding, k)

        results: List[Dict[str, Any]] = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                meta = metadata[idx] if idx < len(metadata) else {"page": None, "source": "Guía turística"}
                results.append(
                    {
                        "chunk": chunks[idx],
                        "distance": float(distances[0][i]),
                        "index": int(idx),
                        "page": meta.get("page"),
                        "source": meta.get("source", "Guía turística"),
                    }
                )

        logger.info(f"Encontrados {len(results)} resultados relevantes")
        return results

    except Exception as e:
        logger.error(f"Error en búsqueda: {e}")
        return []

# 6. Pipeline completo RAG con metadatos para citacion.

def build_rag_pipeline(pdf_path: str) -> Dict[str, Any]:
    """Construye el pipeline completo RAG con metadatos para citación."""
    logger.info(f"Iniciando pipeline RAG para: {pdf_path}")

    pages = load_pdf(pdf_path)
    chunks, metadata = chunk_text(pages)
    embeddings = embed_chunks(chunks)
    index = build_faiss_index(embeddings)

    logger.info("Pipeline RAG completado exitosamente")

    return {
        "pages": pages,
        "chunks": chunks,
        "metadata": metadata,
        "index": index,
        "embeddings": embeddings,
    }
