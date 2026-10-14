#!/usr/bin/env python3
"""
Comprehensive Memory Systems Benchmark for AgriMemory Dataset v2

Benchmarks 5 Memory Systems:
- NMS (Neuroscientific Memory System) - SQLite-based semantic memory
- RAG (Retrieval-Augmented Generation) - Chunked document retrieval
- Hybrid (NMS + RAG) - Combined with Reciprocal Rank Fusion
- BM25 (BM25 + Semantic Reranker) - Lightweight lexical + semantic
- MemoryGraph (Graph-based Memory) - Entity-relation extraction

Evaluates on:
- AgriConvMem (conversational memory with QA by category)
- AgriMultiHop (multi-hop reasoning)

Metrics:
- F1 Score, Exact Match, Semantic Similarity, BLEU-1

Outputs: JSON results + CSV summary files
"""

import argparse
import asyncio
import csv
import json
import logging
import math
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import tiktoken
from dotenv import load_dotenv
from jinja2 import Template
from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm

load_dotenv()

# =============================================================================
# Logging Setup
# =============================================================================
def setup_logging(log_dir: str = "logs", experiment_name: str = "comprehensive") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")

    logger = logging.getLogger("comprehensive_benchmark")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    ))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info(f"Logging to: {log_file}")
    return logger


# =============================================================================
# Prompts
# =============================================================================
ANSWER_PROMPT = """You are an agricultural expert assistant with access to conversation memories.

# Context (Retrieved Memories):
{{CONTEXT}}

# Question:
{{QUESTION}}

# Instructions:
1. Answer based ONLY on the provided context
2. For temporal questions, use dates from the context
3. Keep your answer SHORT and PRECISE (max 5-6 words)
4. If the answer is not in the context, say "unknown"

Answer:"""

LLM_JUDGE_SYSTEM = "You are an expert grader evaluating agricultural domain answers."

LLM_JUDGE_PROMPT = """Evaluate if the generated answer is correct compared to the gold answer.

Question: {question}
Gold Answer: {gold_answer}
Generated Answer: {response}

Guidelines:
- Be generous: if the answer captures the key information, mark as CORRECT
- For disease names, treatments, or technical terms, allow synonyms
- For dates/times, flexible matching (e.g., "last week" vs specific date is OK if contextually correct)
- Partial matches that contain the key information are CORRECT

Return ONLY a JSON object: {{"label": "CORRECT"}} or {{"label": "WRONG"}}"""


# =============================================================================
# Metrics
# =============================================================================
def normalize_answer(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.split())


def calculate_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(normalize_answer(prediction).split())
    ref_tokens = set(normalize_answer(reference).split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(ref_tokens) if ref_tokens else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def calculate_exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def calculate_bleu1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = set(normalize_answer(reference).split())
    if not pred_tokens:
        return 0.0
    matches = sum(1 for t in pred_tokens if t in ref_tokens)
    return matches / len(pred_tokens)


# =============================================================================
# Base Memory System
# =============================================================================
class BaseMemorySystem(ABC):
    def __init__(self, logger: logging.Logger, name: str):
        self.logger = logger
        self.name = name

        # LLM Configuration
        self.chutes_api_key = os.getenv("CHUTES_API_KEY")
        self.chutes_api_base = os.getenv("CHUTES_API_BASE", "https://llm.chutes.ai/v1")
        self.model = os.getenv("MODEL", "gpt-4o-mini")

        # Initialize OpenAI client (for legacy or Chutes.ai compatible endpoint)
        if self.chutes_api_key:
            # Use Chutes.ai API (OpenAI compatible)
            self.openai_client = OpenAI(
                api_key=self.chutes_api_key,
                base_url=self.chutes_api_base
            )
        else:
            # Fallback to OpenAI
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Embedding Configuration
        self.embedding_api_base = os.getenv("EMBEDDING_API_BASE", "")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    @abstractmethod
    def reset(self, user_id: str = None):
        pass

    @abstractmethod
    def add_conversation(self, conversation: List[Dict], metadata: Dict = None):
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> Tuple[str, float]:
        pass

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text using configured endpoint."""
        text = text[:8000]  # Truncate to max input length

        if self.embedding_api_base:
            # Use custom embedding endpoint (Infinity/TEI format)
            try:
                response = requests.post(
                    f"{self.embedding_api_base.rstrip('/')}/embeddings",
                    json={
                        "model": self.embedding_model,
                        "input": text
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                return data["data"][0]["embedding"]
            except Exception as e:
                self.logger.error(f"Custom embedding API error: {e}")
                raise
        else:
            # Use OpenAI client (works with OpenAI or compatible APIs)
            try:
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                return response.data[0].embedding
            except Exception as e:
                self.logger.error(f"OpenAI embedding error: {e}")
                raise

    def generate_response(self, question: str, context: str) -> Tuple[str, float]:
        template = Template(ANSWER_PROMPT)
        prompt = template.render(CONTEXT=context, QUESTION=question)
        try:
            start = time.time()
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=100
            )
            return response.choices[0].message.content.strip(), time.time() - start
        except Exception as e:
            self.logger.error(f"LLM error: {e}")
            return "error", 0

    def query(self, question: str, top_k: int = 10) -> Tuple[str, str, float, float]:
        context, search_time = self.search(question, top_k)
        if not context:
            return "unknown", "", search_time, 0
        answer, llm_time = self.generate_response(question, context)
        return answer, context, search_time, llm_time


# =============================================================================
# Mem0 Cloud Memory System
# =============================================================================
class Mem0CloudSystem(BaseMemorySystem):
    """Mem0 Cloud API with optional graph memory."""

    def __init__(self, logger: logging.Logger, enable_graph: bool = False):
        super().__init__(logger, f"mem0{'_graph' if enable_graph else ''}")
        self.enable_graph = enable_graph
        self.user_id = None

        try:
            from mem0 import MemoryClient
            self.client = MemoryClient(
                api_key=os.getenv("MEM0_API_KEY"),
                org_id=os.getenv("MEM0_ORGANIZATION_ID"),
                project_id=os.getenv("MEM0_PROJECT_ID"),
            )
            logger.info(f"Mem0 Cloud initialized (graph={enable_graph})")
        except Exception as e:
            logger.error(f"Failed to initialize Mem0: {e}")
            raise

    def reset(self, user_id: str = None):
        self.user_id = user_id or f"agri_mem0_{uuid.uuid4().hex[:8]}"
        try:
            self.client.delete_all(user_id=self.user_id)
        except Exception as e:
            self.logger.debug(f"Mem0 delete (may not exist): {e}")

    def add_conversation(self, conversation: List[Dict], metadata: Dict = None):
        timestamp = metadata.get("timestamp", "") if metadata else ""
        messages = []
        for turn in conversation:
            speaker = turn.get("speaker", "user")
            text = turn.get("text", "")
            role = "user" if speaker.lower() in ["farmer", "user"] else "assistant"
            content = f"{timestamp}: {speaker}: {text}" if timestamp else f"{speaker}: {text}"
            messages.append({"role": role, "content": content})

        if messages:
            try:
                self.client.add(
                    messages,
                    user_id=self.user_id,
                    metadata={"timestamp": timestamp},
                    enable_graph=self.enable_graph
                )
            except Exception as e:
                self.logger.warning(f"Mem0 add error: {e}")

    def search(self, query: str, top_k: int = 10) -> Tuple[str, float]:
        start = time.time()
        try:
            if self.enable_graph:
                results = self.client.search(
                    query,
                    user_id=self.user_id,
                    limit=top_k,
                    enable_graph=True,
                    output_format="v1.1"
                )
                memories = []
                if isinstance(results, dict):
                    for mem in results.get("results", []):
                        memories.append(mem.get("memory", ""))
                    for rel in results.get("relations", []):
                        memories.append(f"{rel.get('source')} -> {rel.get('relationship')} -> {rel.get('target')}")
                context = "\n".join(memories)
            else:
                results = self.client.search(
                    query,
                    filters={"user_id": self.user_id},
                    limit=top_k
                )
                if isinstance(results, dict) and "results" in results:
                    results = results["results"]
                memories = [r.get("memory", "") for r in results if isinstance(r, dict)]
                context = "\n".join(memories)

            return context, time.time() - start
        except Exception as e:
            self.logger.error(f"Mem0 search error: {e}")
            return "", time.time() - start


# =============================================================================
# Zep Cloud Memory System
# =============================================================================
class ZepCloudSystem(BaseMemorySystem):
    """Zep Cloud API with knowledge graph."""

    def __init__(self, logger: logging.Logger):
        super().__init__(logger, "zep")
        self.user_id = None
        self.session_id = None

        try:
            from zep_cloud.client import Zep
            from zep_cloud import Message
            self.client = Zep(api_key=os.getenv("ZEP_API_KEY"))
            self.Message = Message
            logger.info("Zep Cloud initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Zep: {e}")
            raise

    def reset(self, user_id: str = None):
        self.user_id = user_id or f"agri_zep_{uuid.uuid4().hex[:8]}"
        self.session_id = f"thread_{self.user_id}"
        try:
            self.client.user.add(user_id=self.user_id)
        except Exception as e:
            self.logger.debug(f"Zep user exists: {e}")
        try:
            # Use thread.create for Zep Cloud v3 API
            self.client.thread.create(thread_id=self.session_id, user_id=self.user_id)
        except Exception as e:
            self.logger.debug(f"Zep thread exists: {e}")

    def add_conversation(self, conversation: List[Dict], metadata: Dict = None):
        timestamp = metadata.get("timestamp", "") if metadata else ""
        for turn in conversation:
            speaker = turn.get("speaker", "user")
            text = turn.get("text", "")
            content = f"{timestamp}: {text}" if timestamp else text
            try:
                # Use thread.add_messages for Zep Cloud v3 API
                self.client.thread.add_messages(
                    thread_id=self.session_id,
                    messages=[self.Message(role=speaker, role_type="user", content=content)]
                )
            except Exception as e:
                self.logger.warning(f"Zep add error: {e}")

    def search(self, query: str, top_k: int = 10) -> Tuple[str, float]:
        start = time.time()
        try:
            # Search edges (facts)
            edges_result = self.client.graph.search(
                user_id=self.user_id,
                query=query,
                scope="edges",
                limit=top_k
            )
            # Search nodes (entities)
            nodes_result = self.client.graph.search(
                user_id=self.user_id,
                query=query,
                scope="nodes",
                limit=top_k
            )

            context_parts = []
            if hasattr(edges_result, 'edges') and edges_result.edges:
                for edge in edges_result.edges:
                    fact = edge.fact if hasattr(edge, 'fact') else str(edge)
                    context_parts.append(f"FACT: {fact}")

            if hasattr(nodes_result, 'nodes') and nodes_result.nodes:
                for node in nodes_result.nodes:
                    name = node.name if hasattr(node, 'name') else str(node)
                    summary = node.summary if hasattr(node, 'summary') else ""
                    context_parts.append(f"ENTITY: {name} - {summary}")

            return "\n".join(context_parts), time.time() - start
        except Exception as e:
            self.logger.error(f"Zep search error: {e}")
            return "", time.time() - start


# =============================================================================
# RAG Baseline (Local with OpenAI embeddings)
# =============================================================================
class RAGSystem(BaseMemorySystem):
    """RAG with OpenAI embeddings and chunking."""

    def __init__(self, logger: logging.Logger, chunk_size: int = 500, top_k: int = 5):
        super().__init__(logger, "rag")
        self.chunk_size = chunk_size
        self.default_top_k = top_k
        self.chunks = []
        self.embeddings = []
        self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        logger.info(f"RAG system initialized (chunk_size={chunk_size})")

    def reset(self, user_id: str = None):
        self.chunks = []
        self.embeddings = []

    def add_conversation(self, conversation: List[Dict], metadata: Dict = None):
        timestamp = metadata.get("timestamp", "") if metadata else ""
        conv_text = ""
        for turn in conversation:
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            conv_text += f"{timestamp} | {speaker}: {text}\n"

        # Chunk and embed
        tokens = self.encoding.encode(conv_text)
        for i in range(0, len(tokens), self.chunk_size):
            chunk = self.encoding.decode(tokens[i:i + self.chunk_size])
            try:
                emb = self.get_embedding(chunk)
                self.chunks.append(chunk)
                self.embeddings.append(emb)
            except Exception as e:
                self.logger.warning(f"RAG embedding error: {e}")

    def search(self, query: str, top_k: int = None) -> Tuple[str, float]:
        if not self.chunks:
            return "", 0
        k = top_k or self.default_top_k
        start = time.time()
        try:
            query_emb = self.get_embedding(query)
            similarities = [
                np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
                for emb in self.embeddings
            ]
            top_idx = np.argsort(similarities)[-k:][::-1]
            context = "\n---\n".join([self.chunks[i] for i in top_idx])
            return context, time.time() - start
        except Exception as e:
            self.logger.error(f"RAG search error: {e}")
            return "", time.time() - start


# =============================================================================
# Full Context Baseline
# =============================================================================
class FullContextBaseline(BaseMemorySystem):
    """No retrieval - passes all context to LLM."""

    def __init__(self, logger: logging.Logger, max_tokens: int = 15000):
        super().__init__(logger, "baseline")
        self.max_tokens = max_tokens
        self.context = ""
        self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        logger.info("Full-context baseline initialized")

    def reset(self, user_id: str = None):
        self.context = ""

    def add_conversation(self, conversation: List[Dict], metadata: Dict = None):
        timestamp = metadata.get("timestamp", "") if metadata else ""
        for turn in conversation:
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            self.context += f"{timestamp} | {speaker}: {text}\n"

    def search(self, query: str, top_k: int = 10) -> Tuple[str, float]:
        tokens = self.encoding.encode(self.context)
        if len(tokens) > self.max_tokens:
            return self.encoding.decode(tokens[:self.max_tokens]), 0
        return self.context, 0


# =============================================================================
# NMS - Neuroscientific Memory System (renamed from OpenMemory)
# =============================================================================
class NMSSystem(BaseMemorySystem):
    """NMS - Neuroscientific Memory System with SQLite storage."""

    def __init__(self, logger: logging.Logger, db_path: str = "./data/nms.db"):
        super().__init__(logger, "nms")
        self.user_id = None
        self.db_path = db_path
        self.conn = None

        import sqlite3
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
        logger.info(f"NMS initialized at {db_path}")

    def _init_db(self):
        """Initialize SQLite database with memories table."""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                timestamp TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON memories(user_id)')
        self.conn.commit()

    def reset(self, user_id: str = None):
        self.user_id = user_id or f"agri_openmem_{uuid.uuid4().hex[:8]}"
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM memories WHERE user_id = ?', (self.user_id,))
        self.conn.commit()

    def add_conversation(self, conversation: List[Dict], metadata: Dict = None):
        timestamp = metadata.get("timestamp", "") if metadata else ""
        cursor = self.conn.cursor()

        for turn in conversation:
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            content = f"{timestamp} | {speaker}: {text}"

            try:
                embedding = self.get_embedding(content)
                embedding_blob = json.dumps(embedding).encode()
                cursor.execute(
                    'INSERT INTO memories (user_id, content, embedding, timestamp, metadata) VALUES (?, ?, ?, ?, ?)',
                    (self.user_id, content, embedding_blob, timestamp, json.dumps(metadata or {}))
                )
            except Exception as e:
                self.logger.warning(f"NMS add error: {e}")

        self.conn.commit()

    def search(self, query: str, top_k: int = 10) -> Tuple[str, float]:
        start = time.time()
        try:
            query_embedding = self.get_embedding(query)

            cursor = self.conn.cursor()
            cursor.execute(
                'SELECT content, embedding FROM memories WHERE user_id = ?',
                (self.user_id,)
            )
            rows = cursor.fetchall()

            if not rows:
                return "", time.time() - start

            # Calculate cosine similarity
            similarities = []
            for content, embedding_blob in rows:
                if embedding_blob:
                    doc_embedding = json.loads(embedding_blob.decode())
                    similarity = np.dot(query_embedding, doc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                    )
                    similarities.append((similarity, content))

            # Sort by similarity and get top_k
            similarities.sort(reverse=True, key=lambda x: x[0])
            top_results = similarities[:top_k]
            context = "\n".join([content for _, content in top_results])

            return context, time.time() - start
        except Exception as e:
            self.logger.error(f"NMS search error: {e}")
            return "", time.time() - start

    def __del__(self):
        if self.conn:
            self.conn.close()


# =============================================================================
# Hybrid NMS + RAG System
# =============================================================================
class HybridNMSRAG(BaseMemorySystem):
    """Combines NMS with RAG using Reciprocal Rank Fusion."""

    def __init__(self, logger: logging.Logger, db_path: str = "./data/hybrid.db"):
        super().__init__(logger, "hybrid")
        import sqlite3

        # Initialize NMS component
        self.user_id = None
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

        # Initialize RAG component
        self.rag_chunks = []
        self.rag_embeddings = []
        self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        self.chunk_size = 500

        logger.info("Hybrid NMS+RAG initialized")

    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                timestamp TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON memories(user_id)')
        self.conn.commit()

    def reset(self, user_id: str = None):
        self.user_id = user_id or f"hybrid_{uuid.uuid4().hex[:8]}"
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM memories WHERE user_id = ?', (self.user_id,))
        self.conn.commit()
        self.rag_chunks = []
        self.rag_embeddings = []

    def add_conversation(self, conversation: List[Dict], metadata: Dict = None):
        timestamp = metadata.get("timestamp", "") if metadata else ""
        cursor = self.conn.cursor()
        conv_text = ""

        for turn in conversation:
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            content = f"{timestamp} | {speaker}: {text}"
            conv_text += content + "\n"

            try:
                embedding = self.get_embedding(content)
                embedding_blob = json.dumps(embedding).encode()
                cursor.execute(
                    'INSERT INTO memories (user_id, content, embedding, timestamp, metadata) VALUES (?, ?, ?, ?, ?)',
                    (self.user_id, content, embedding_blob, timestamp, json.dumps(metadata or {}))
                )
            except Exception as e:
                self.logger.warning(f"Hybrid NMS add error: {e}")

        self.conn.commit()

        # RAG chunking
        tokens = self.encoding.encode(conv_text)
        for i in range(0, len(tokens), self.chunk_size):
            chunk = self.encoding.decode(tokens[i:i + self.chunk_size])
            try:
                emb = self.get_embedding(chunk)
                self.rag_chunks.append(chunk)
                self.rag_embeddings.append(emb)
            except Exception as e:
                self.logger.warning(f"Hybrid RAG embedding error: {e}")

    def _reciprocal_rank_fusion(self, rankings: List[List[Tuple[str, float]]], k: int = 60) -> List[Tuple[str, float]]:
        """Combine multiple rankings using RRF."""
        scores = {}
        for ranking in rankings:
            for rank, (text, _) in enumerate(ranking):
                text_key = text[:200]
                if text_key not in scores:
                    scores[text_key] = {"text": text, "score": 0}
                scores[text_key]["score"] += 1.0 / (k + rank + 1)
        results = [(v["text"], v["score"]) for v in scores.values()]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def search(self, query: str, top_k: int = 10) -> Tuple[str, float]:
        start = time.time()
        try:
            query_emb = self.get_embedding(query)

            # NMS search
            nms_results = []
            cursor = self.conn.cursor()
            cursor.execute('SELECT content, embedding FROM memories WHERE user_id = ?', (self.user_id,))
            rows = cursor.fetchall()
            for content, embedding_blob in rows:
                if embedding_blob:
                    doc_emb = json.loads(embedding_blob.decode())
                    sim = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
                    nms_results.append((content, sim))
            nms_results.sort(key=lambda x: x[1], reverse=True)

            # RAG search
            rag_results = []
            for i, emb in enumerate(self.rag_embeddings):
                sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
                rag_results.append((self.rag_chunks[i], sim))
            rag_results.sort(key=lambda x: x[1], reverse=True)

            # Reciprocal Rank Fusion
            combined = self._reciprocal_rank_fusion([nms_results[:top_k*2], rag_results[:top_k*2]])
            context = "\n---\n".join([text for text, _ in combined[:top_k]])

            return context, time.time() - start
        except Exception as e:
            self.logger.error(f"Hybrid search error: {e}")
            return "", time.time() - start

    def __del__(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()


# =============================================================================
# BM25 + Semantic Reranker System
# =============================================================================
class BM25Reranker(BaseMemorySystem):
    """BM25 retrieval with semantic reranking - lightweight approach."""

    def __init__(self, logger: logging.Logger, top_k_bm25: int = 20, top_k_final: int = 5):
        super().__init__(logger, "bm25")
        self.documents = []
        self.doc_tokens = []
        self.doc_embeddings = []
        self.df = defaultdict(int)
        self.avgdl = 0
        self.k1 = 1.5
        self.b = 0.75
        self.top_k_bm25 = top_k_bm25
        self.top_k_final = top_k_final
        logger.info("BM25+Reranker initialized")

    def reset(self, user_id: str = None):
        self.documents = []
        self.doc_tokens = []
        self.doc_embeddings = []
        self.df = defaultdict(int)
        self.avgdl = 0

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    def add_conversation(self, conversation: List[Dict], metadata: Dict = None):
        timestamp = metadata.get("timestamp", "") if metadata else ""
        for turn in conversation:
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            content = f"{timestamp} | {speaker}: {text}"

            tokens = self._tokenize(content)
            self.documents.append(content)
            self.doc_tokens.append(tokens)

            for token in set(tokens):
                self.df[token] += 1

            try:
                emb = self.get_embedding(content)
                self.doc_embeddings.append(emb)
            except Exception as e:
                self.doc_embeddings.append(None)
                self.logger.warning(f"BM25 embedding error: {e}")

        if self.doc_tokens:
            self.avgdl = sum(len(d) for d in self.doc_tokens) / len(self.doc_tokens)

    def _bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        score = 0
        doc_tokens = self.doc_tokens[doc_idx]
        doc_len = len(doc_tokens)
        N = len(self.documents)

        for term in query_tokens:
            if term not in self.df:
                continue
            df = self.df[term]
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            term_tf = doc_tokens.count(term)
            numerator = term_tf * (self.k1 + 1)
            denominator = term_tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl + 1e-9))
            score += idf * numerator / (denominator + 1e-9)

        return score

    def search(self, query: str, top_k: int = None) -> Tuple[str, float]:
        if not self.documents:
            return "", 0
        start = time.time()
        k = top_k or self.top_k_final

        try:
            query_tokens = self._tokenize(query)

            # BM25 retrieval
            bm25_scores = [(i, self._bm25_score(query_tokens, i)) for i in range(len(self.documents))]
            bm25_scores.sort(key=lambda x: x[1], reverse=True)
            candidates = bm25_scores[:self.top_k_bm25]

            # Semantic reranking
            query_emb = self.get_embedding(query)
            reranked = []
            for doc_idx, bm25_score in candidates:
                doc_emb = self.doc_embeddings[doc_idx]
                if doc_emb:
                    sem_score = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
                    combined = 0.4 * bm25_score + 0.6 * sem_score
                else:
                    combined = bm25_score
                reranked.append((doc_idx, combined))

            reranked.sort(key=lambda x: x[1], reverse=True)
            context = "\n---\n".join([self.documents[i] for i, _ in reranked[:k]])

            return context, time.time() - start
        except Exception as e:
            self.logger.error(f"BM25 search error: {e}")
            return "", time.time() - start


# =============================================================================
# MemoryGraph System (Graphiti-inspired)
# =============================================================================
class MemoryGraph(BaseMemorySystem):
    """Graph-based memory with entity extraction - inspired by Graphiti/MemGPT."""

    def __init__(self, logger: logging.Logger, db_path: str = "./data/memorygraph.db"):
        super().__init__(logger, "memorygraph")
        import sqlite3

        self.user_id = None
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
        logger.info("MemoryGraph initialized")

    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                entity_type TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                source TEXT NOT NULL,
                relation TEXT NOT NULL,
                target TEXT NOT NULL,
                fact TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                timestamp TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_user ON entities(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_relation_user ON relations(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_episode_user ON episodes(user_id)')
        self.conn.commit()

    def reset(self, user_id: str = None):
        self.user_id = user_id or f"graph_{uuid.uuid4().hex[:8]}"
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM entities WHERE user_id = ?', (self.user_id,))
        cursor.execute('DELETE FROM relations WHERE user_id = ?', (self.user_id,))
        cursor.execute('DELETE FROM episodes WHERE user_id = ?', (self.user_id,))
        self.conn.commit()

    def _extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract entities using regex patterns for agricultural domain."""
        entities = []

        disease_patterns = [
            r'\b(late blight|early blight|powdery mildew|downy mildew|rust|anthracnose|'
            r'bacterial wilt|fusarium wilt|root rot|leaf spot|mosaic virus|blight)\b'
        ]
        crop_patterns = [
            r'\b(tomato|potato|rice|wheat|corn|maize|soybean|cotton|sugarcane|'
            r'apple|grape|citrus|mango|banana|coffee|tea|pepper|onion|garlic)\b'
        ]
        treatment_patterns = [
            r'\b(fungicide|pesticide|herbicide|neem oil|copper sulfate|'
            r'organic treatment|chemical spray|biological control)\b'
        ]

        for pattern in disease_patterns:
            for match in re.finditer(pattern, text.lower()):
                entities.append((match.group(), "disease"))

        for pattern in crop_patterns:
            for match in re.finditer(pattern, text.lower()):
                entities.append((match.group(), "crop"))

        for pattern in treatment_patterns:
            for match in re.finditer(pattern, text.lower()):
                entities.append((match.group(), "treatment"))

        return list(set(entities))

    def add_conversation(self, conversation: List[Dict], metadata: Dict = None):
        timestamp = metadata.get("timestamp", "") if metadata else ""
        cursor = self.conn.cursor()

        for turn in conversation:
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            content = f"{timestamp} | {speaker}: {text}"

            try:
                content_emb = self.get_embedding(content)
                emb_blob = json.dumps(content_emb).encode()

                cursor.execute(
                    'INSERT INTO episodes (user_id, content, embedding, timestamp) VALUES (?, ?, ?, ?)',
                    (self.user_id, content, emb_blob, timestamp)
                )

                entities = self._extract_entities(text)
                for entity_name, entity_type in entities:
                    entity_emb = self.get_embedding(entity_name)
                    cursor.execute(
                        'INSERT INTO entities (user_id, name, entity_type, embedding) VALUES (?, ?, ?, ?)',
                        (self.user_id, entity_name, entity_type, json.dumps(entity_emb).encode())
                    )

                if len(entities) >= 2:
                    for i, (e1, t1) in enumerate(entities):
                        for e2, t2 in entities[i+1:]:
                            fact = f"{e1} ({t1}) is related to {e2} ({t2})"
                            fact_emb = self.get_embedding(fact)
                            cursor.execute(
                                'INSERT INTO relations (user_id, source, relation, target, fact, embedding) VALUES (?, ?, ?, ?, ?, ?)',
                                (self.user_id, e1, "related_to", e2, fact, json.dumps(fact_emb).encode())
                            )
            except Exception as e:
                self.logger.warning(f"MemoryGraph add error: {e}")

        self.conn.commit()

    def search(self, query: str, top_k: int = 10) -> Tuple[str, float]:
        start = time.time()
        try:
            query_emb = self.get_embedding(query)
            query_entities = self._extract_entities(query)

            results = []
            cursor = self.conn.cursor()

            # Entity-based search
            for entity_name, _ in query_entities:
                cursor.execute(
                    'SELECT DISTINCT r.fact, r.embedding FROM relations r WHERE r.user_id = ? AND (r.source = ? OR r.target = ?)',
                    (self.user_id, entity_name, entity_name)
                )
                for fact, emb_blob in cursor.fetchall():
                    if emb_blob:
                        doc_emb = json.loads(emb_blob.decode())
                        sim = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
                        results.append((fact, sim + 0.1))

            # Semantic search on episodes
            cursor.execute('SELECT content, embedding FROM episodes WHERE user_id = ?', (self.user_id,))
            for content, emb_blob in cursor.fetchall():
                if emb_blob:
                    doc_emb = json.loads(emb_blob.decode())
                    sim = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
                    results.append((content, sim))

            # Deduplicate and sort
            seen = set()
            unique_results = []
            for text, score in results:
                key = text[:100]
                if key not in seen:
                    seen.add(key)
                    unique_results.append((text, score))

            unique_results.sort(key=lambda x: x[1], reverse=True)
            context = "\n---\n".join([text for text, _ in unique_results[:top_k]])

            return context, time.time() - start
        except Exception as e:
            self.logger.error(f"MemoryGraph search error: {e}")
            return "", time.time() - start

    def __del__(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()


# =============================================================================
# Memory System Factory
# =============================================================================
def create_memory_system(technique: str, logger: logging.Logger) -> Optional[BaseMemorySystem]:
    try:
        if technique == "nms":
            return NMSSystem(logger)
        elif technique == "rag":
            return RAGSystem(logger, chunk_size=500, top_k=5)
        elif technique == "hybrid":
            return HybridNMSRAG(logger)
        elif technique == "bm25":
            return BM25Reranker(logger)
        elif technique == "memorygraph":
            return MemoryGraph(logger)
        elif technique == "mem0":
            return Mem0CloudSystem(logger, enable_graph=False)
        elif technique == "mem0_graph":
            return Mem0CloudSystem(logger, enable_graph=True)
        elif technique == "zep":
            return ZepCloudSystem(logger)
        elif technique == "rag_large":
            return RAGSystem(logger, chunk_size=2000, top_k=3)
        elif technique == "baseline":
            return FullContextBaseline(logger)
        else:
            logger.warning(f"Unknown technique: {technique}")
            return None
    except Exception as e:
        logger.error(f"Failed to create {technique}: {e}")
        return None


# =============================================================================
# LLM Judge
# =============================================================================
async def llm_judge(client: AsyncOpenAI, question: str, gold: str, response: str) -> bool:
    prompt = LLM_JUDGE_PROMPT.format(question=question, gold_answer=gold, response=response)
    try:
        result = await client.chat.completions.create(
            model=os.getenv("EVAL_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": LLM_JUDGE_SYSTEM},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        content = result.choices[0].message.content
        match = re.search(r'\{\s*"label"\s*:\s*"(\w+)"\s*\}', content)
        if match:
            return match.group(1).lower() == "correct"
        return False
    except Exception as e:
        print(f"LLM judge error: {e}")
        return False


# =============================================================================
# Data Loading
# =============================================================================
def load_agri_locomo(data_path: str) -> List[Dict]:
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data.get("samples", data) if isinstance(data, dict) else data


def load_agri_hotpotqa(data_path: str) -> List[Dict]:
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("samples", [])


# =============================================================================
# Evaluation Functions
# =============================================================================
async def evaluate_locomo(
    data_path: str,
    memory_system: BaseMemorySystem,
    logger: logging.Logger,
    top_k: int = 10,
    limit: int = None,
    use_llm_judge: bool = True
) -> Dict[str, Any]:
    logger.info(f"Evaluating LoCoMo with {memory_system.name}")
    samples = load_agri_locomo(data_path)
    if limit:
        samples = samples[:limit]

    async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    results = {
        "predictions": [],
        "f1": [],
        "em": [],
        "bleu1": [],
        "llm_judge": [],
        "by_category": defaultdict(list)
    }

    for sample_idx, sample in enumerate(tqdm(samples, desc=f"LoCoMo ({memory_system.name})")):
        sample_id = sample.get("sample_id", f"sample_{sample_idx}")
        memory_system.reset(f"locomo_{sample_id}")

        # Build memory
        conversation = sample.get("conversation", {})
        for key, value in conversation.items():
            if key.startswith("session_") and isinstance(value, list):
                timestamp = conversation.get(f"{key}_date_time", "")
                memory_system.add_conversation(value, {"timestamp": timestamp})

        await asyncio.sleep(0.1)  # Rate limiting

        # Evaluate QA
        for qa in sample.get('qa', []):
            question = qa["question"]
            reference = str(qa["answer"])
            category = qa.get("category", 0)
            category_name = qa.get("category_name", str(category))

            prediction, context, search_time, llm_time = memory_system.query(question, top_k)

            f1 = calculate_f1(prediction, reference)
            em = calculate_exact_match(prediction, reference)
            bleu = calculate_bleu1(prediction, reference)

            llm_correct = False
            if use_llm_judge:
                llm_correct = await llm_judge(async_client, question, reference, prediction)

            results["predictions"].append({
                "sample_id": sample_id,
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "category": category,
                "category_name": category_name,
                "f1": f1,
                "em": em,
                "bleu1": bleu,
                "llm_judge": llm_correct,
                "search_time": search_time,
                "llm_time": llm_time
            })
            results["f1"].append(f1)
            results["em"].append(em)
            results["bleu1"].append(bleu)
            results["llm_judge"].append(1 if llm_correct else 0)
            results["by_category"][category_name].append(f1)

    return {
        "dataset": "AgriLoCoMo",
        "technique": memory_system.name,
        "num_samples": len(samples),
        "num_questions": len(results["f1"]),
        "overall": {
            "f1": np.mean(results["f1"]) if results["f1"] else 0,
            "em": np.mean(results["em"]) if results["em"] else 0,
            "bleu1": np.mean(results["bleu1"]) if results["bleu1"] else 0,
            "llm_judge": np.mean(results["llm_judge"]) if results["llm_judge"] else 0
        },
        "by_category": {cat: np.mean(scores) for cat, scores in results["by_category"].items()},
        "predictions": results["predictions"]
    }


async def evaluate_hotpotqa(
    data_path: str,
    memory_system: BaseMemorySystem,
    logger: logging.Logger,
    top_k: int = 10,
    limit: int = None,
    use_llm_judge: bool = True
) -> Dict[str, Any]:
    logger.info(f"Evaluating HotpotQA with {memory_system.name}")
    samples = load_agri_hotpotqa(data_path)
    if limit:
        samples = samples[:limit]

    async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    results = {
        "predictions": [],
        "f1": [],
        "em": [],
        "llm_judge": [],
        "by_type": defaultdict(list)
    }

    for sample_idx, sample in enumerate(tqdm(samples, desc=f"HotpotQA ({memory_system.name})")):
        sample_id = sample.get("id", f"hotpot_{sample_idx}")
        memory_system.reset(f"hotpot_{sample_id}")

        # Build memory from context
        for para in sample.get("context", []):
            title = para.get("title", "")
            text = para.get("text", "")
            memory_system.add_conversation(
                [{"speaker": "document", "text": f"{title}: {text}" if title else text}],
                {"title": title}
            )

        await asyncio.sleep(0.1)

        question = sample.get("question", "")
        reference = str(sample.get("answer", ""))
        q_type = sample.get("type", "bridge")

        prediction, context, search_time, llm_time = memory_system.query(question, top_k)

        f1 = calculate_f1(prediction, reference)
        em = calculate_exact_match(prediction, reference)

        llm_correct = False
        if use_llm_judge:
            llm_correct = await llm_judge(async_client, question, reference, prediction)

        results["predictions"].append({
            "sample_id": sample_id,
            "question": question,
            "reference": reference,
            "prediction": prediction,
            "type": q_type,
            "f1": f1,
            "em": em,
            "llm_judge": llm_correct
        })
        results["f1"].append(f1)
        results["em"].append(em)
        results["llm_judge"].append(1 if llm_correct else 0)
        results["by_type"][q_type].append(f1)

    return {
        "dataset": "AgriHotpotQA",
        "technique": memory_system.name,
        "num_samples": len(samples),
        "overall": {
            "f1": np.mean(results["f1"]) if results["f1"] else 0,
            "em": np.mean(results["em"]) if results["em"] else 0,
            "llm_judge": np.mean(results["llm_judge"]) if results["llm_judge"] else 0
        },
        "by_type": {qtype: np.mean(scores) for qtype, scores in results["by_type"].items()},
        "predictions": results["predictions"]
    }


def print_results(results: Dict, logger: logging.Logger):
    output = [
        "\n" + "=" * 60,
        f"  {results['dataset']} - {results['technique']}",
        "=" * 60,
        f"Samples: {results['num_samples']}"
    ]
    if "num_questions" in results:
        output.append(f"Questions: {results['num_questions']}")

    output.append("\nOverall Metrics:")
    for metric, value in results["overall"].items():
        output.append(f"  {metric}: {value:.4f}")

    if results.get("by_category"):
        output.append("\nBy Category (F1):")
        for cat, score in sorted(results["by_category"].items()):
            output.append(f"  {cat}: {score:.4f}")

    if results.get("by_type"):
        output.append("\nBy Type (F1):")
        for qtype, score in sorted(results["by_type"].items()):
            output.append(f"  {qtype}: {score:.4f}")

    output.append("=" * 60)
    for line in output:
        print(line)
        logger.info(line)


# =============================================================================
# Main
# =============================================================================
async def main_async():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Comprehensive Memory Systems Benchmark")
    parser.add_argument("--dataset", choices=["locomo", "hotpotqa", "both"], default="both")
    parser.add_argument("--data-dir", default="data/text")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="results/comprehensive_benchmark.json")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--techniques", default="nms,rag,hybrid,bm25,memorygraph",
                       help="Comma-separated: nms,rag,hybrid,bm25,memorygraph,mem0,zep,baseline")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-llm-judge", action="store_true")
    parser.add_argument("--experiment-name", default="comprehensive")

    args = parser.parse_args()

    logger = setup_logging(args.log_dir, args.experiment_name)
    logger.info(f"Starting benchmark: {vars(args)}")

    techniques = [t.strip() for t in args.techniques.split(",")]
    logger.info(f"Techniques: {techniques}")

    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)

    all_results = {}

    for technique in techniques:
        logger.info(f"\n{'='*60}\nBenchmarking: {technique}\n{'='*60}")

        memory_system = create_memory_system(technique, logger)
        if not memory_system:
            continue

        technique_results = {}

        # LoCoMo
        if args.dataset in ["locomo", "both"]:
            locomo_path = Path(args.data_dir) / "agri_locomo_v2" / f"{args.split}.json"
            if locomo_path.exists():
                results = await evaluate_locomo(
                    str(locomo_path), memory_system, logger,
                    args.top_k, args.limit, not args.no_llm_judge
                )
                technique_results["locomo"] = results
                print_results(results, logger)

        # HotpotQA
        if args.dataset in ["hotpotqa", "both"]:
            hotpotqa_path = Path(args.data_dir) / "agri_hotpotqa_v4" / f"{args.split}.json"
            if hotpotqa_path.exists():
                results = await evaluate_hotpotqa(
                    str(hotpotqa_path), memory_system, logger,
                    args.top_k, args.limit, not args.no_llm_judge
                )
                technique_results["hotpotqa"] = results
                print_results(results, logger)

        all_results[technique] = technique_results

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    logger.info(f"\nResults saved to: {output_path}")

    # Export to CSV
    csv_dir = output_path.parent

    # Summary CSV
    summary_csv = csv_dir / "summary.csv"
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['technique', 'locomo_f1', 'locomo_em', 'locomo_bleu1', 'locomo_llm_judge',
                        'hotpotqa_f1', 'hotpotqa_em', 'hotpotqa_llm_judge'])
        for technique, tech_results in all_results.items():
            loc = tech_results.get("locomo", {}).get("overall", {})
            hot = tech_results.get("hotpotqa", {}).get("overall", {})
            writer.writerow([
                technique,
                loc.get("f1", 0), loc.get("em", 0), loc.get("bleu1", 0), loc.get("llm_judge", 0),
                hot.get("f1", 0), hot.get("em", 0), hot.get("llm_judge", 0)
            ])
    logger.info(f"Summary CSV saved to: {summary_csv}")

    # LoCoMo by category CSV
    locomo_csv = csv_dir / "locomo_by_category.csv"
    with open(locomo_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['technique', 'category', 'f1'])
        for technique, tech_results in all_results.items():
            for cat, score in tech_results.get("locomo", {}).get("by_category", {}).items():
                writer.writerow([technique, cat, score])
    logger.info(f"LoCoMo by category CSV saved to: {locomo_csv}")

    # HotpotQA by type CSV
    hotpotqa_csv = csv_dir / "hotpotqa_by_type.csv"
    with open(hotpotqa_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['technique', 'question_type', 'f1'])
        for technique, tech_results in all_results.items():
            for qtype, score in tech_results.get("hotpotqa", {}).get("by_type", {}).items():
                writer.writerow([technique, qtype, score])
    logger.info(f"HotpotQA by type CSV saved to: {hotpotqa_csv}")

    # Print summary table
    print("\n" + "=" * 100)
    print("  COMPREHENSIVE BENCHMARK SUMMARY")
    print("=" * 100)
    print(f"\n{'Technique':<15} {'LoCoMo F1':<12} {'LoCoMo EM':<12} {'LoCoMo LLM':<12} {'HotpotQA F1':<12} {'HotpotQA LLM':<12}")
    print("-" * 100)

    for technique, tech_results in all_results.items():
        loc_f1 = tech_results.get("locomo", {}).get("overall", {}).get("f1", 0)
        loc_em = tech_results.get("locomo", {}).get("overall", {}).get("em", 0)
        loc_llm = tech_results.get("locomo", {}).get("overall", {}).get("llm_judge", 0)
        hot_f1 = tech_results.get("hotpotqa", {}).get("overall", {}).get("f1", 0)
        hot_llm = tech_results.get("hotpotqa", {}).get("overall", {}).get("llm_judge", 0)
        print(f"{technique:<15} {loc_f1:<12.4f} {loc_em:<12.4f} {loc_llm:<12.4f} {hot_f1:<12.4f} {hot_llm:<12.4f}")

    print("=" * 100)
    print(f"\nCSV files saved to: {csv_dir}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
