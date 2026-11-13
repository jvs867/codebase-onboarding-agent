"""Codebase Onboarding Agent: Streamlit app.

Ask plain-English questions about any public Python GitHub repository. The app indexes the
repository's docs and code in Pinecone (RAG), then a LangGraph agent explores the repository
with seven tools, cites `file:line` for every claim and checks each citation before answering.

This is the notebook (Codebase_Onboarding_Project1-sears.ipynb) as a single runnable file,
with the Gradio UI replaced by Streamlit.

Run it:
    pip install -r requirements.txt
    export GOOGLE_API_KEY=...  PINECONE_API_KEY=...      (or enter them in the sidebar)
    streamlit run app.py

Optional environment variables (or entries in .streamlit/secrets.toml):
    UI_USERNAME, UI_PASSWORD   require a login before the app can be used
    CLONE_DIR                  where repositories are cloned (default: ./repos)
    CHAT_MODEL                 first Gemini chat model to try
    PINECONE_INDEX_NAME        Pinecone index to use (default: codebase-onboarding)
"""

from __future__ import annotations

import ast
import hashlib
import hmac
import os
import random
import re
import shutil
import subprocess
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Annotated, TypedDict

import streamlit as st
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pinecone import Pinecone, ServerlessSpec


# ---------------------------------------------------------------------------------------
# 1 · Settings
# ---------------------------------------------------------------------------------------

def setting(name: str, default: str | None = None) -> str | None:
    """Read a setting from the environment, then from Streamlit secrets."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        value = st.secrets.get(name)      # raises if there is no secrets file
    except Exception:
        value = None
    return str(value) if value else default


# Models
CHAT_MODEL = setting("CHAT_MODEL", "gemini-3.8-flash")
# Tried in order if CHAT_MODEL is unavailable or rate-limited
FALLBACK_CHAT_MODELS = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
THINKING_LEVEL = "low"          # Gemini 3+ only
MODEL_TIMEOUT_SECONDS = 300     # per call
PROBE_TIMEOUT_SECONDS = 20      # per model, for the startup check
EMBEDDING_MODEL = "models/gemini-embedding-001"

# Pinecone
PINECONE_INDEX_NAME = setting("PINECONE_INDEX_NAME", "codebase-onboarding")   # lowercase, digits, '-'
PINECONE_CLOUD = "aws"          # free-plan region
PINECONE_REGION = "us-east-1"

CLONE_DIR = setting("CLONE_DIR", "repos")

# Indexing
SKIP_DIRS = {".git", ".github", "node_modules", "venv", ".venv", "env", "build", "dist",
             "__pycache__", "site-packages", ".tox", ".nox", ".mypy_cache", ".pytest_cache",
             "_build", "htmlcov", "migrations", "vendor", "third_party"}
INCLUDE_TESTS = False
DOC_EXTENSIONS = {".md", ".rst"}  # plus .txt inside docs/ folders
DOC_SKIP_NAMES = {"changelog", "changes", "history", "license", "licence", "authors", "code_of_conduct"}
MAX_FILE_BYTES = 300_000

# Chunking (characters)
DOC_CHUNK_SIZE = 2000
DOC_CHUNK_OVERLAP = 200
MAX_CODE_CHUNK_CHARS = 6000      # Pinecone metadata limit is ~40KB

# Rate limiting / cost
MAX_CHUNKS_PER_REPO = 2500       # protects embedding quota

# Embedding rate limits for gemini-embedding-001 (check yours in AI Studio > Rate limits).
# Free tier: 100 requests/min, 30,000 tokens/min, 1,000 requests/day.
EMBED_TPM_LIMIT = 30_000
EMBED_RPM_LIMIT = 100
EMBED_LIMIT_SAFETY = 0.8         # use 80% of the limits (Google measures a rolling minute)
EMBED_MAX_BATCH_TOKENS = 6_000   # tokens per request
EMBED_MAX_BATCH_TEXTS = 100      # API maximum per request
CHARS_PER_TOKEN = 3.0            # conservative estimate (code tokenises densely)
PINECONE_UPSERT_BATCH = 20       # vectors per Pinecone upsert (2 MB request limit)

# Agent
SEARCH_TOP_K = 6
AGENT_RECURSION_LIMIT = 40       # ~2 steps per tool call
MAX_TOOL_CALLS = 15              # hard budget per question; the prompt asks for 4-10
MAX_REVISIONS = 1                # times the citation check may send an answer back

# UI
ALLOWED_REPOS = None             # e.g. ["psf/requests"]; None allows any

EXAMPLE_QUESTIONS = [
    "Give me a high-level tour: what does this project do and how is the code organized?",
    "What are the main entry points a user of this library calls?",
    "Which files should a new contributor read first, and why?",
    "Where are errors and exceptions defined, and when are they raised?",
]

SYSTEM_PROMPT = """You are a senior engineer onboarding a new contributor to a Python codebase.
Answer questions about the active repository by exploring it with your tools. Never guess.

Research strategy:
1. If the user gives a repository URL that isn't the active repository, call load_repository first.
2. Get the big picture with search_docs (and list_files when the structure matters).
3. Find entry points with search_code. Prefer the search tools over reading whole files:
   read_file only the line ranges that search_code / find_symbol / grep_repo point you to.
4. Follow the trail: when code calls or imports a name you need to understand, use find_symbol,
   then read_file to read it. Use grep_repo to find exact strings or every call site.
5. Stop exploring when you can explain the flow end to end. Usually 4-10 tool calls are enough;
   never more than 15. A high-level tour does not need every file read in full.
6. Do not make up information. If you do not know say so.
7. If the request is unclear, ask the user for clarification.

Answer format (Markdown):
- **Summary**: 2-3 sentences.
- **Walkthrough**: numbered steps in the order the code actually runs. Cite every step as
  `path/to/file.py:START-END` (the full path from the repository root), and include short code
  snippets where they help.
- **Key files**: a bullet list of the files a newcomer should read first.
If something isn't in the repository, say so plainly instead of inventing it."""


# ---------------------------------------------------------------------------------------
# 2 · Clone and filter the repository
# ---------------------------------------------------------------------------------------

def parse_github_url(url: str):
    """Extract (owner, repo) from a GitHub URL or 'owner/repo'."""
    url = url.strip().rstrip("/")
    m = re.search(r"github\.com[/:]([^/\s]+)/([^/\s#?]+)", url) or re.fullmatch(r"([\w.-]+)/([\w.-]+)", url)
    if not m:
        raise ValueError(f"Not a recognizable GitHub repository URL: {url}")
    owner, repo = m.group(1), m.group(2)
    repo = repo[:-4] if repo.endswith(".git") else repo
    return owner, repo


def repo_namespace(owner: str, repo: str) -> str:
    """Pinecone namespace name for a repo, e.g. 'pallets--itsdangerous'."""
    return re.sub(r"[^a-z0-9-]", "-", f"{owner}--{repo}".lower())


def clone_repository(owner: str, repo: str) -> Path:
    """Shallow-clone into CLONE_DIR (once) and return the path."""
    dest = Path(CLONE_DIR) / f"{owner}__{repo}"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning https://github.com/{owner}/{repo} ...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", f"https://github.com/{owner}/{repo}.git", str(dest)],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"git clone failed (is the repo public and the URL right?):\n{result.stderr}")
    return dest


def _is_test_path(rel_parts) -> bool:
    name = rel_parts[-1]
    return (any(p in {"tests", "test", "testing"} for p in rel_parts[:-1])
            or name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py")


def iter_repo_files(repo_path: Path):
    """Yield (relative_path, kind) for indexable files; kind is 'code' or 'doc'."""
    for root, dirs, files in os.walk(repo_path):
        # Prune in place so os.walk skips these
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        for fname in sorted(files):
            abs_path = Path(root) / fname
            rel = abs_path.relative_to(repo_path)
            if abs_path.stat().st_size > MAX_FILE_BYTES:
                continue
            if not INCLUDE_TESTS and _is_test_path(rel.parts):
                continue
            suffix, stem = abs_path.suffix.lower(), abs_path.stem.lower()
            if suffix == ".py":
                yield rel.as_posix(), "code"
            elif stem in DOC_SKIP_NAMES:
                continue
            elif suffix in DOC_EXTENSIONS or (suffix == ".txt" and "doc" in str(rel.parent).lower()
                                              and not stem.startswith("requirements")):
                yield rel.as_posix(), "doc"


# ---------------------------------------------------------------------------------------
# 3 · Chunk documentation (by heading) and Python code (by AST node)
# ---------------------------------------------------------------------------------------

md_header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False)
size_splitter = RecursiveCharacterTextSplitter(chunk_size=DOC_CHUNK_SIZE, chunk_overlap=DOC_CHUNK_OVERLAP)


def chunk_doc_file(rel_path: str, text: str) -> list[Document]:
    """Split one documentation file into heading-aware chunks."""
    if rel_path.lower().endswith(".md"):
        sections = md_header_splitter.split_text(text)
    else:
        sections = [Document(page_content=text, metadata={})]   # .rst/.txt: size split only
    chunks = []
    for piece in size_splitter.split_documents(sections):
        heading = " > ".join(piece.metadata[h] for h in ("h1", "h2", "h3") if h in piece.metadata)
        chunks.append(Document(
            page_content=f"Document: {rel_path}\nSection: {heading or '(top)'}\n\n{piece.page_content}",
            metadata={"kind": "doc", "path": rel_path, "section": heading or "(top)"},
        ))
    return chunks


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n# ... (truncated, use read_file for the rest)"


def chunk_python_file(rel_path: str, source: str):
    """Return (chunks, symbols) for one Python file."""
    lines = source.splitlines()
    module_name = rel_path[:-3].replace("/", ".").removesuffix(".__init__")
    chunks, symbols = [], []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Unparseable (e.g. Python 2): use 80-line chunks
        for start in range(0, len(lines), 80):
            block = lines[start:start + 80]
            chunks.append(Document(
                page_content=f"File: {rel_path}\nLines: {start + 1}-{start + len(block)}\n\n" + "\n".join(block),
                metadata={"kind": "code", "path": rel_path, "start_line": start + 1,
                          "end_line": start + len(block), "symbol": rel_path, "symbol_type": "lines"}))
        return chunks, symbols

    def span(node):
        """Line range, including decorators."""
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        return start, node.end_lineno

    def signature(node) -> str:
        """def/class header as a single line."""
        out = []
        for line in lines[node.lineno - 1: node.end_lineno]:
            out.append(line.strip())
            if line.rstrip().endswith(":"):
                break
        return " ".join(out)[:300]

    def add_chunk(qualname, symbol_type, start, end, sig, docstring, body):
        header = [f"File: {rel_path}", f"{symbol_type.title()}: {qualname}",
                  f"Lines: {start}-{end}", f"Signature: {sig}"]
        if docstring:
            header.append("Docstring: " + docstring[:600])
        chunks.append(Document(
            page_content=_truncate("\n".join(header) + "\n\n" + body, MAX_CODE_CHUNK_CHARS),
            metadata={"kind": "code", "path": rel_path, "start_line": start, "end_line": end,
                      "symbol": qualname, "symbol_type": symbol_type}))
        symbols.append({"name": qualname.split(".")[-1], "qualname": qualname, "type": symbol_type,
                        "path": rel_path, "start_line": start, "end_line": end, "signature": sig})

    def add_function(node, qualname, symbol_type):
        start, end = span(node)
        add_chunk(qualname, symbol_type, start, end, signature(node),
                  ast.get_docstring(node) or "", "\n".join(lines[start - 1:end]))

    defined, imports, other_top_level = [], [], []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add_function(node, node.name, "function")
            defined.append(node.name)
        elif isinstance(node, ast.ClassDef):
            defined.append(node.name)
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            # Class overview: everything except methods and docstring
            class_level = [ast.get_source_segment(source, n) or "" for n in node.body
                           if n not in methods and not (isinstance(n, ast.Expr)
                                                        and isinstance(getattr(n, "value", None), ast.Constant))]
            overview = signature(node) + "\n"
            overview += "Methods: " + ", ".join(m.name for m in methods) + "\n"
            overview += "\n".join(s for s in class_level if s)
            start, end = span(node)
            add_chunk(node.name, "class", start, end, signature(node), ast.get_docstring(node) or "", overview)
            for m in methods:
                add_function(m, f"{node.name}.{m.name}", "method")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.get_source_segment(source, node) or "")
        else:
            other_top_level.append(ast.get_source_segment(source, node) or "")

    # Module overview: docstring, imports, top-level names and code
    mod_doc = ast.get_docstring(tree) or ""
    body = ("Imports:\n" + "\n".join(imports) + "\n\nDefines: " + ", ".join(defined) +
            "\n\nOther top-level code:\n" + "\n".join(s for s in other_top_level if s))
    add_chunk(module_name, "module", 1, max(len(lines), 1), f"module {module_name}", mod_doc, body)
    return chunks, symbols


def build_chunks(repo_path: Path):
    """Returns (doc_chunks, code_chunks, symbols, file_counts)."""
    doc_chunks, code_chunks, symbols = [], [], []
    counts = {"doc": 0, "code": 0}
    for rel_path, kind in iter_repo_files(repo_path):
        text = (repo_path / rel_path).read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        counts[kind] += 1
        if kind == "doc":
            doc_chunks.extend(chunk_doc_file(rel_path, text))
        else:
            c, s = chunk_python_file(rel_path, text)
            code_chunks.extend(c)
            symbols.extend(s)
    return doc_chunks, code_chunks, symbols, counts


# ---------------------------------------------------------------------------------------
# 4 · Embedding helpers (pacing for the Gemini free tier)
# ---------------------------------------------------------------------------------------

def _field(obj, key, default=None):
    """Read a field from a Pinecone response (dict or object)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _chunk_id(namespace: str, d: Document) -> str:
    """Deterministic ID so re-indexing overwrites instead of duplicating."""
    return hashlib.md5(f"{namespace}|{d.metadata['path']}|{d.metadata.get('symbol', '')}|"
                       f"{d.metadata.get('start_line', '')}|{d.page_content[:200]}".encode()).hexdigest()


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


class EmbeddingRateLimiter:
    """Keeps the last 60 seconds of embedding calls under the token and request budgets.
    Waits *before* each call, so the API is never asked for more than the quota allows."""

    def __init__(self, tpm: int, rpm: int, safety: float):
        self.base_tpm = tpm * safety
        self.tpm = self.base_tpm
        self.rpm = max(1, int(rpm * safety))
        self.calls = deque()                      # (timestamp, tokens)

    def _prune(self, now):
        while self.calls and now - self.calls[0][0] >= 60:
            self.calls.popleft()

    def wait_for(self, tokens: int):
        tokens = min(tokens, self.tpm)            # an oversized call still runs, on an empty window
        while True:
            now = time.time()
            self._prune(now)
            used = sum(t for _, t in self.calls)
            if used + tokens <= self.tpm and len(self.calls) < self.rpm:
                self.calls.append((now, tokens))
                return
            # Sleep until enough of the oldest calls leave the 60-second window
            freed, wait = 0, 1.0
            for i, (t0, tok) in enumerate(self.calls):
                freed += tok
                if used - freed + tokens <= self.tpm and len(self.calls) - (i + 1) < self.rpm:
                    wait = t0 + 60 - now
                    break
            time.sleep(max(wait, 0.5))

    def penalize(self, seconds: float):
        """After a 429: block for `seconds`, then run 20% slower (our token estimate may be low)."""
        self.tpm = max(self.tpm * 0.8, self.base_tpm * 0.3)
        self.calls.clear()
        self.calls.append((time.time() - 60 + seconds, self.tpm))

    def eta_minutes(self, tokens: int) -> float:
        return tokens / self.tpm


class DailyQuotaExceeded(RuntimeError):
    pass


def _retry_after(msg: str):
    """Seconds Google asks us to wait ('retryDelay': '37s' or 'Please retry in 37.2s')."""
    m = re.search(r"retry(?:Delay)?['\"]?\s*(?::|in)\s*['\"]?(\d+(?:\.\d+)?)s", msg)
    return float(m.group(1)) if m else None


def _token_batches(chunks: list[Document]):
    """Group chunks into requests of at most EMBED_MAX_BATCH_TOKENS / EMBED_MAX_BATCH_TEXTS."""
    batch, tokens = [], 0
    for d in chunks:
        t = estimate_tokens(d.page_content)
        if batch and (tokens + t > EMBED_MAX_BATCH_TOKENS or len(batch) >= EMBED_MAX_BATCH_TEXTS):
            yield batch, tokens
            batch, tokens = [], 0
        batch.append(d)
        tokens += t
    if batch:
        yield batch, tokens


# ---------------------------------------------------------------------------------------
# 5 · Small text helpers shared by the agent and the UI
# ---------------------------------------------------------------------------------------

def _text(content) -> str:
    """Text from a message's content (a string or a list of parts)."""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)


def _is_capacity_error(text: str) -> bool:
    return any(k in text for k in ("429", "503", "504", "DEADLINE", "UNAVAILABLE",
                                   "RESOURCE_EXHAUSTED", "quota", "imeout"))


def _probe_failure_reason(e: Exception) -> str:
    s = str(e)
    return ("busy right now (503), try again later" if "503" in s or "UNAVAILABLE" in s
            else "timed out on Google's side (504), usually overload" if "504" in s or "DEADLINE" in s
            else "rate-limited (429)" if "429" in s or "RESOURCE_EXHAUSTED" in s
            else "not available to your key (404)" if "404" in s or "NOT_FOUND" in s
            else s[:150])


# `path/to/file.py:12-34` or `path/to/file.py`; only file-like extensions, so names such as
# `Signer.sign` or `json.dumps` aren't mistaken for files.
CITE_RE = re.compile(r"`([\w./-]+\.(?:py|pyi|md|rst|txt|toml|cfg|ini|ya?ml|json))(?::(\d+)(?:-(\d+))?)?`")
# Same idea, any extension: used to turn citations in the final answer into GitHub links.
CITATION_RE = re.compile(r"(?<!\[)`([\w./-]+\.[A-Za-z0-9]+)(?::(\d+)(?:-(\d+))?)?`")


def find_citations(text: str):
    """[(path, start, end)] for every `file:line` citation in the text (start/end may be None)."""
    out = []
    for m in CITE_RE.finditer(text):
        start = int(m.group(2)) if m.group(2) else None
        end = int(m.group(3)) if m.group(3) else start
        out.append((m.group(1).lstrip("/"), start, end))
    return out


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]   # the conversation (kept per thread)
    tool_calls: int    # tool calls used on the current question
    revisions: int     # citation-check rewrites on the current question
    check_note: str    # result of the last citation check, shown in the trace


EMPTY_REPO = dict(url=None, owner=None, repo=None, path=None, namespace=None, symbols={}, summary=None)


# ---------------------------------------------------------------------------------------
# 6 · Backend: Gemini + Pinecone connections, indexing, tools and the LangGraph agent
#
# Streamlit reruns this script on every interaction, so everything that must outlive a rerun
# (model clients, the Pinecone index, loaded repositories, conversation memory) lives in one
# Backend object that st.cache_resource creates once and shares between sessions.
# ---------------------------------------------------------------------------------------

class Backend:
    def __init__(self, google_api_key: str, pinecone_api_key: str, log=print):
        self.google_api_key = google_api_key
        self.lock = threading.Lock()       # one question / one indexing job at a time (shared state)
        self.repo_cache: dict[str, dict] = {}                 # repos loaded this run, by namespace
        self.state: dict = dict(EMPTY_REPO, symbols=defaultdict(list))   # active repo, used by tools
        self.rate_limiter = EmbeddingRateLimiter(EMBED_TPM_LIMIT, EMBED_RPM_LIMIT, EMBED_LIMIT_SAFETY)
        self._connect_models(log)
        self._connect_pinecone(pinecone_api_key, log)
        self.tools = self._make_tools()
        self.agent = self._build_agent()

    # ---- connections -------------------------------------------------------------------
    def _make_llm(self, model_name: str, timeout: int = MODEL_TIMEOUT_SECONDS,
                  max_retries: int = 2) -> ChatGoogleGenerativeAI:
        # timeout/max_retries stop a hung or overloaded call from blocking forever
        extra = {"thinking_level": THINKING_LEVEL} if model_name.startswith("gemini-3") and THINKING_LEVEL else {}
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=self.google_api_key,
                                      timeout=timeout, max_retries=max_retries, **extra)

    def _connect_models(self, log):
        """Probe each chat model; those that respond form the fallback chain."""
        self.working_models, failures = [], []
        for name in [CHAT_MODEL] + [m for m in FALLBACK_CHAT_MODELS if m != CHAT_MODEL]:
            candidate = self._make_llm(name)
            log(f"Checking {name}...")
            try:
                # Short timeout and no retries here, so a hung model can't stall startup for minutes
                self._make_llm(name, timeout=PROBE_TIMEOUT_SECONDS, max_retries=0).invoke("Reply with OK.")
                self.working_models.append((name, candidate))
                log(f"✅ {name} is available")
            except Exception as e:
                failures.append(f"{name}: {_probe_failure_reason(e)}")
                log(f"⚠️ {name}: {_probe_failure_reason(e)}")
        if not self.working_models:
            raise RuntimeError(
                "None of the Gemini chat models responded:\n- " + "\n- ".join(failures) +
                "\n\nIf they were busy (503), wait a few minutes and try again. Otherwise check the "
                "GOOGLE_API_KEY, or set CHAT_MODEL to a model your key can use.")
        self.model_name = self.working_models[0][0]
        self.llm = self.working_models[0][1]
        self.fallback_llms = [m for _, m in self.working_models[1:]]
        self.embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=self.google_api_key)
        # Detect the dimension so the embedding model can be swapped
        self.embedding_dim = len(self.embeddings.embed_query("dimension check"))

    def _connect_pinecone(self, pinecone_api_key: str, log):
        self.pc = Pinecone(api_key=pinecone_api_key)
        # Pinecone allows only lowercase letters, digits and '-'
        self.index_name = re.sub(r"[^a-z0-9-]+", "-", PINECONE_INDEX_NAME.lower()).strip("-")[:45]
        if not self.pc.has_index(self.index_name):
            log(f"Creating Pinecone index '{self.index_name}' (takes about a minute)...")
            self.pc.create_index(name=self.index_name, dimension=self.embedding_dim, metric="cosine",
                                 spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION))
        self.index = self.pc.Index(self.index_name)
        existing_dim = self.pc.describe_index(self.index_name).dimension
        if existing_dim != self.embedding_dim:
            raise RuntimeError(
                f"Pinecone index '{self.index_name}' has dimension {existing_dim}, but the embedding model "
                f"returns {self.embedding_dim}. Set PINECONE_INDEX_NAME to a new name to create a fresh index.")

    def vector_store(self, namespace: str) -> PineconeVectorStore:
        """A LangChain vector store for one repository's namespace."""
        return PineconeVectorStore(index=self.index, embedding=self.embeddings, namespace=namespace)

    # ---- indexing ----------------------------------------------------------------------
    def indexed_namespaces(self) -> dict[str, int]:
        """{namespace: vector count} for every repo indexed in Pinecone."""
        namespaces = _field(self.index.describe_index_stats(), "namespaces", {}) or {}
        return {ns: int(_field(info, "vector_count", 0) or 0) for ns, info in namespaces.items()}

    def _existing_ids(self, namespace: str, ids: list[str]) -> set:
        """IDs from this batch that are already stored in Pinecone."""
        try:
            res = self.index.fetch(ids=ids, namespace=namespace)
            return set((_field(res, "vectors", {}) or {}).keys())
        except Exception:
            return set()

    def _embed_with_retry(self, texts: list[str], tokens: int) -> list[list[float]]:
        """One paced embedding request, retried on rate-limit and transient errors."""
        for attempt in range(8):
            self.rate_limiter.wait_for(tokens)
            try:
                return self.embeddings.embed_documents(texts, batch_size=EMBED_MAX_BATCH_TEXTS)
            except Exception as e:
                msg = str(e)
                if "PerDay" in msg or "per day" in msg.lower():
                    raise DailyQuotaExceeded(
                        "Gemini's daily embedding quota is used up. Everything embedded so far is saved: "
                        "load the repo again after the quota resets (midnight Pacific) and it will resume, "
                        "or enable billing in AI Studio.") from e
                transient = any(k in msg for k in ("429", "RESOURCE_EXHAUSTED", "500", "503", "504",
                                                   "UNAVAILABLE", "DEADLINE", "INTERNAL"))
                if not transient or attempt == 7:
                    raise
                wait = (_retry_after(msg) or min(60, 5 * 2 ** attempt)) + random.uniform(1, 3)
                self.rate_limiter.penalize(wait)
                print(f"   ⏳ Gemini rate limit hit, waiting {wait:.0f}s "
                      f"(pacing now {self.rate_limiter.tpm:,.0f} tokens/min)")

    def _upsert_with_retry(self, namespace: str, records: list[dict]):
        for i in range(0, len(records), PINECONE_UPSERT_BATCH):
            part = records[i:i + PINECONE_UPSERT_BATCH]
            for attempt in range(5):
                try:
                    self.index.upsert(vectors=part, namespace=namespace)
                    break
                except Exception as e:
                    if attempt == 4:
                        raise
                    print(f"   ⚠️  Pinecone: {type(e).__name__}: {str(e)[:120]} — retrying")
                    time.sleep(2 * 2 ** attempt)

    def upload_chunks(self, namespace: str, chunks: list[Document], progress=None, skip_existing=False):
        """Embed chunks (paced to the Gemini quota) and upsert them into Pinecone.
        skip_existing: don't re-embed chunks already in Pinecone (resumes an interrupted upload).
        progress: optional callback(fraction, message)."""
        total_tokens = sum(estimate_tokens(d.page_content) for d in chunks)
        remaining, done, skipped = total_tokens, 0, 0
        for batch, tokens in _token_batches(chunks):
            ids = [_chunk_id(namespace, d) for d in batch]
            present = self._existing_ids(namespace, ids) if skip_existing else set()
            todo = [(d, i) for d, i in zip(batch, ids) if i not in present]
            if todo:
                todo_tokens = sum(estimate_tokens(d.page_content) for d, _ in todo)
                vectors = self._embed_with_retry([d.page_content for d, _ in todo], todo_tokens)
                self._upsert_with_retry(namespace, [
                    {"id": i, "values": v, "metadata": {**d.metadata, "text": d.page_content}}
                    for (d, i), v in zip(todo, vectors)])
            done += len(batch)
            skipped += len(present)
            remaining -= tokens
            eta = self.rate_limiter.eta_minutes(remaining)
            if progress:
                progress(0.05 + 0.95 * done / len(chunks),
                         f"Embedding chunks {done}/{len(chunks)} (~{eta:.0f} min left)")

    def prepare_repository(self, repo_url: str, force_reindex: bool = False, progress=None) -> str:
        """Clone, chunk, embed (if needed) and activate a repo. Returns a summary."""
        owner, repo = parse_github_url(repo_url)
        if ALLOWED_REPOS is not None and f"{owner}/{repo}".lower() not in {r.lower() for r in ALLOWED_REPOS}:
            raise ValueError(f"{owner}/{repo} is not in ALLOWED_REPOS. Allowed: {', '.join(ALLOWED_REPOS)}")
        namespace = repo_namespace(owner, repo)

        # Reuse if already loaded this run
        cached = self.repo_cache.get(namespace)
        if cached and not force_reindex and Path(cached["path"]).exists():
            self.state.update(cached)
            return cached["summary"]

        if progress:
            progress(0.02, f"Cloning {owner}/{repo}...")
        if force_reindex:   # get a fresh clone
            shutil.rmtree(Path(CLONE_DIR) / f"{owner}__{repo}", ignore_errors=True)
        repo_path = clone_repository(owner, repo)
        if progress:
            progress(0.05, "Splitting docs and code into chunks...")

        doc_chunks, code_chunks, symbols, counts = build_chunks(repo_path)
        all_chunks = doc_chunks + code_chunks
        total_chunks, dropped = len(all_chunks), 0
        if total_chunks > MAX_CHUNKS_PER_REPO:
            # Keep docs, then module/class overviews (broad coverage), then functions/methods
            code_sorted = sorted(code_chunks, key=lambda d: d.metadata["symbol_type"] not in ("module", "class"))
            all_chunks = (doc_chunks + code_sorted)[:MAX_CHUNKS_PER_REPO]
            dropped = total_chunks - len(all_chunks)

        if force_reindex:
            try:
                self.index.delete(delete_all=True, namespace=namespace)
            except Exception:
                pass   # namespace may not exist

        existing = 0 if force_reindex else self.indexed_namespaces().get(namespace, 0)
        if existing < len(all_chunks):
            # existing > 0 means a previous upload was interrupted: finish it instead of trusting it
            self.upload_chunks(namespace, all_chunks, progress=progress, skip_existing=existing > 0)
            time.sleep(3)   # Pinecone is eventually consistent

        # Build the symbol table and activate the repo
        table = defaultdict(list)
        for s in symbols:
            table[s["name"].lower()].append(s)
            if s["qualname"].lower() != s["name"].lower():
                table[s["qualname"].lower()].append(s)
        top_level = sorted(p.name + ("/" if p.is_dir() else "") for p in repo_path.iterdir()
                           if not p.name.startswith("."))
        summary = (f"Repository {owner}/{repo} is loaded and searchable.\n"
                   f"Indexed: {counts['doc']} documentation files, {counts['code']} Python files, "
                   f"{len(symbols)} symbols"
                   + (f" (⚠️ {dropped} of {total_chunks} chunks not embedded: over MAX_CHUNKS_PER_REPO)"
                      if dropped else "")
                   + f".\nTop-level contents: {', '.join(top_level)}")
        repo_state = dict(url=f"https://github.com/{owner}/{repo}", owner=owner, repo=repo,
                          path=repo_path, namespace=namespace, symbols=table, summary=summary)
        self.repo_cache[namespace] = repo_state
        self.state.update(repo_state)
        if progress:
            progress(1.0, "Ready")
        return summary

    def delete_repository_index(self, namespace: str):
        """Remove a repo's vectors from Pinecone and forget it."""
        self.index.delete(delete_all=True, namespace=namespace)
        self.repo_cache.pop(namespace, None)
        if self.state["namespace"] == namespace:
            self.activate(None)

    def activate(self, namespace: str | None):
        """Point the tools at one session's repository (call while holding self.lock)."""
        if namespace in self.repo_cache:
            self.state.update(self.repo_cache[namespace])
        else:
            self.state.update(dict(EMPTY_REPO, symbols=defaultdict(list)))

    # ---- paths and citations -----------------------------------------------------------
    def _require_repo(self):
        if self.state["path"] is None:
            raise ValueError("No repository loaded. Call load_repository with a GitHub URL first.")

    def _safe_path(self, rel_path: str) -> Path:
        """Resolve a repo-relative path; reject paths outside the repo."""
        self._require_repo()
        root = self.state["path"].resolve()
        full = (root / rel_path.strip().lstrip("/")).resolve()
        if root != full and root not in full.parents:
            raise ValueError("Path is outside the repository.")
        return full

    def citation_problems(self, text: str):
        """Check every citation against the cloned repo. Returns (problems, number_of_citations)."""
        cites = find_citations(text)
        problems, seen = [], set()
        for path, start, end in cites:
            if (path, start, end) in seen:
                continue
            seen.add((path, start, end))
            try:
                full = self._safe_path(path)
            except Exception:
                problems.append(f"`{path}` is outside the repository")
                continue
            if not full.is_file():
                similar = sorted(p.relative_to(self.state["path"]).as_posix()
                                 for p in self.state["path"].rglob(Path(path).name))[:3]
                hint = f" (did you mean {', '.join(f'`{s}`' for s in similar)}?)" if similar else ""
                problems.append(f"`{path}` does not exist{hint}")
            elif start is not None:
                n_lines = len(full.read_text(encoding="utf-8", errors="ignore").splitlines())
                if start < 1 or end < start or start > n_lines or end > n_lines + 2:
                    problems.append(f"`{path}:{start}-{end}` is outside the file ({n_lines} lines)")
        if not cites and "walkthrough" in text.lower():
            problems.append("the walkthrough has no `file:START-END` citations")
        return problems, len(cites)

    def link_citations(self, text: str) -> str:
        """Turn file:line citations into GitHub links (only for files that really exist)."""
        if not self.state["url"]:
            return text

        def repl(m):
            path, start, end = m.group(1).lstrip("/"), m.group(2), m.group(3) or m.group(2)
            try:
                if not self._safe_path(path).is_file():
                    return m.group(0)
            except Exception:
                return m.group(0)
            anchor = f"#L{start}-L{end}" if start else ""
            return f"[{m.group(0)}]({self.state['url']}/blob/HEAD/{path}{anchor})"
        return CITATION_RE.sub(repl, text)

    # ---- the seven tools ---------------------------------------------------------------
    def _vector_search(self, query: str, kind: str) -> str:
        self._require_repo()
        hits = self.vector_store(self.state["namespace"]).similarity_search_with_score(
            query, k=SEARCH_TOP_K, filter={"kind": kind})
        if not hits:
            return "No results."
        out = []
        for doc, score in hits:
            m = doc.metadata
            where = (f"{m['path']} > {m.get('section', '')}" if kind == "doc"
                     else f"{m['path']}:{int(m.get('start_line', 0))}-{int(m.get('end_line', 0))} "
                          f"({m.get('symbol_type')} {m.get('symbol')})")
            out.append(f"### {where}  [score {score:.3f}]\n{doc.page_content[:1500]}")
        return "\n\n".join(out)

    def _make_tools(self):
        backend = self

        @tool
        def load_repository(repo_url: str) -> str:
            """Load a public GitHub repository so the other tools can search it. Call this whenever the user
            mentions a repository URL that isn't already the active repository. The first load of a repo can
            take a few minutes because it is embedded; later loads are quick."""
            try:
                return backend.prepare_repository(repo_url)
            except Exception as e:
                return f"Error loading repository: {e}"

        @tool
        def search_docs(query: str) -> str:
            """Semantic search over the repository's documentation (README, docs/ folder, guides).
            Best for high-level questions: what the project does, setup, configuration, concepts, design."""
            try:
                return backend._vector_search(query, "doc")
            except Exception as e:
                return f"Error: {e}"

        @tool
        def search_code(query: str) -> str:
            """Semantic search over the repository's Python code (functions, classes, methods, modules).
            Describe the behavior you're looking for in plain English, e.g. 'validate a signed token'.
            Results include file paths and line ranges."""
            try:
                return backend._vector_search(query, "code")
            except Exception as e:
                return f"Error: {e}"

        @tool
        def find_symbol(name: str) -> str:
            """Look up the definition of a function, class, method or module by exact name, e.g. 'loads',
            'Signer', or 'Signer.sign'. Use this to follow names you saw in code. Returns file and lines."""
            try:
                backend._require_repo()
                table = backend.state["symbols"]
                key = name.strip().lower()
                matches = table.get(key, [])
                if not matches:   # fall back to partial matches
                    matches = [s for k, lst in table.items() if key in k for s in lst]
                seen, lines = set(), []
                for s in matches:
                    ident = (s["path"], s["start_line"])
                    if ident in seen:
                        continue
                    seen.add(ident)
                    lines.append(f"- {s['type']} {s['qualname']} — {s['path']}:{s['start_line']}-{s['end_line']}"
                                 f"  `{s['signature']}`")
                return "\n".join(lines[:25]) or f"No symbol named '{name}' found."
            except Exception as e:
                return f"Error: {e}"

        @tool
        def grep_repo(pattern: str, file_glob: str = "*.py") -> str:
            """Exact text / regular-expression search across files (default: Python files). Use it for exact
            strings: call sites like 'verify_token(', config keys, error messages. file_glob can be '*.md', '*' etc."""
            try:
                backend._require_repo()
            except Exception as e:
                return f"Error: {e}"
            try:
                regex = re.compile(pattern)
            except re.error:
                regex = re.compile(re.escape(pattern))   # invalid regex: match literally
            root = backend.state["path"]
            results = []
            try:
                files = sorted(root.rglob(file_glob))
            except Exception as e:
                return f"Error: bad file_glob: {e}"
            for f in files:
                rel_parts = f.relative_to(root).parts
                if not f.is_file() or any(p in SKIP_DIRS or p.startswith(".") for p in rel_parts[:-1]):
                    continue
                try:
                    for n, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                        if regex.search(line):
                            results.append(f"{'/'.join(rel_parts)}:{n}: {line.strip()[:160]}")
                            if len(results) >= 40:
                                return "\n".join(results) + "\n... (more matches, narrow the pattern)"
                except OSError:
                    continue
            return "\n".join(results) or "No matches."

        @tool
        def read_file(path: str, start_line: int = 1, end_line: int = 200) -> str:
            """Read lines from a file in the repository (path relative to the repo root). At most 250 lines
            per call; use the line ranges from other tools to read exactly what you need."""
            try:
                full = backend._safe_path(path)
                file_lines = full.read_text(encoding="utf-8", errors="ignore").splitlines()
                start = max(1, int(start_line))
                end = min(len(file_lines), int(end_line), start + 249)
                numbered = [f"{n:>5} | {file_lines[n - 1]}" for n in range(start, end + 1)]
                return f"{path} (lines {start}-{end} of {len(file_lines)})\n" + "\n".join(numbered)
            except Exception as e:
                return f"Error: {e}"

        @tool
        def list_files(directory: str = "") -> str:
            """List files and folders (two levels deep) under a directory of the repository.
            Use '' for the repository root."""
            try:
                base = backend._safe_path(directory)
                out = []
                for p in sorted(base.rglob("*")):
                    rel = p.relative_to(base)
                    if len(rel.parts) > 2 or any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts):
                        continue
                    out.append(("  " * (len(rel.parts) - 1)) + rel.name + ("/" if p.is_dir() else ""))
                    if len(out) >= 200:
                        out.append("... (truncated)")
                        break
                return "\n".join(out) or "(empty)"
            except Exception as e:
                return f"Error: {e}"

        return [load_repository, search_docs, search_code, find_symbol, grep_repo, read_file, list_files]

    # ---- the LangGraph agent -----------------------------------------------------------
    #
    #   START ─► agent ──(tool calls)──► tools ──► agent ─► ...
    #              │
    #              └──(final answer)──► check_citations ──(problems found)──► agent  (rewrite, once)
    #                                          └──(citations OK)──► END
    def _build_agent(self):
        tools = self.tools

        def with_fallbacks(primary, backups):
            """On an error, retry the same call with each fallback model in turn."""
            return primary.with_fallbacks(backups) if backups else primary

        def no_tools(model):
            """The model with tools declared but switched off, used once the budget is spent."""
            try:
                return model.bind_tools(tools, tool_choice="none")
            except Exception:
                return model

        model_with_tools = with_fallbacks(self.llm.bind_tools(tools),
                                          [m.bind_tools(tools) for m in self.fallback_llms])
        model_without_tools = with_fallbacks(no_tools(self.llm), [no_tools(m) for m in self.fallback_llms])

        def agent_node(state: AgentState) -> dict:
            """Decide on the next tool call(s), or write the answer."""
            used = state.get("tool_calls", 0)
            messages = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
            if used >= MAX_TOOL_CALLS:
                messages.append(HumanMessage(
                    f"[Tool budget of {MAX_TOOL_CALLS} calls reached. Write the final answer now from what "
                    "you have found, and say what you could not check. Do not call any more tools.]"))
                reply = model_without_tools.invoke(messages)
                if getattr(reply, "tool_calls", None):    # the model ignored the instruction
                    reply = AIMessage(content=_text(reply.content) or
                                      "I reached my tool-call budget before I could finish. "
                                      "Try a narrower question, or ask a follow-up.")
            else:
                reply = model_with_tools.invoke(messages)
            return {"messages": [reply], "tool_calls": used + len(getattr(reply, "tool_calls", None) or [])}

        def check_citations(state: AgentState) -> dict:
            """Verify the answer's citations; send it back once if any are wrong."""
            if self.state["path"] is None:
                return {"check_note": "skipped (no repository loaded)"}
            answer = _text(state["messages"][-1].content)
            problems, n = self.citation_problems(answer)
            if not problems:
                return {"check_note": f"✅ {n} citation(s) verified against the repository" if n
                        else "✅ no file citations to verify"}
            if state.get("revisions", 0) >= MAX_REVISIONS:
                return {"check_note": "⚠️ problems remain after a rewrite: " + "; ".join(problems)}
            feedback = ("[Automatic citation check] Some citations in your answer don't match the repository:\n- "
                        + "\n- ".join(problems)
                        + "\nUse your tools to find the right files and line ranges, then rewrite the whole "
                          "answer in the same format. Drop any claim you can't support with a real "
                          "`file:START-END`.")
            return {"messages": [HumanMessage(feedback)],
                    "revisions": state.get("revisions", 0) + 1,
                    "check_note": "🔁 sent back for a rewrite: " + "; ".join(problems)}

        def route_after_agent(state: AgentState) -> str:
            return "tools" if getattr(state["messages"][-1], "tool_calls", None) else "check_citations"

        def route_after_check(state: AgentState) -> str:
            return "agent" if isinstance(state["messages"][-1], HumanMessage) else END

        builder = StateGraph(AgentState)
        builder.add_node("agent", agent_node)
        builder.add_node("tools", ToolNode(tools))
        builder.add_node("check_citations", check_citations)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", route_after_agent, ["tools", "check_citations"])
        builder.add_edge("tools", "agent")
        builder.add_conditional_edges("check_citations", route_after_check, ["agent", END])
        # The checkpointer remembers each conversation (thread) for follow-up questions
        return builder.compile(checkpointer=InMemorySaver())

    def agent_events(self, question: str, thread_id: str):
        """Run the agent on one question, yielding events:
            {"type": "tool_call",   "id", "name", "args"}
            {"type": "tool_result", "id", "name", "content"}
            {"type": "slow",        "seconds"}           (step took >30s)
            {"type": "check",       "text"}              (citation check result)
            {"type": "answer",      "text", "seconds"}   (last, on success)
            {"type": "error",       "text"}
        """
        message = f"[Active repository: {self.state['url']}]\n\n{question}" if self.state["url"] else question
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": AGENT_RECURSION_LIMIT}
        final_answer, started = "", time.time()
        last = started
        try:
            # Yields each model turn / tool run as it finishes
            for step in self.agent.stream({"messages": [{"role": "user", "content": message}],
                                           "tool_calls": 0, "revisions": 0, "check_note": ""},  # reset per question
                                          config=config, stream_mode="updates"):
                now = time.time()
                if now - last > 30:
                    yield {"type": "slow", "seconds": round(now - last)}
                last = now
                for node, node_output in step.items():
                    if node == "check_citations" and (node_output or {}).get("check_note"):
                        yield {"type": "check", "text": node_output["check_note"]}
                    for msg in (node_output or {}).get("messages", []):
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for call in msg.tool_calls:
                                yield {"type": "tool_call", "id": call.get("id"),
                                       "name": call["name"], "args": call["args"]}
                        elif isinstance(msg, ToolMessage):
                            yield {"type": "tool_result", "id": msg.tool_call_id, "name": msg.name,
                                   "content": _text(msg.content)}
                        elif isinstance(msg, AIMessage):
                            final_answer = _text(msg.content)
        except Exception as e:
            text = f"{type(e).__name__}: {str(e)[:400]}"
            if _is_capacity_error(text):
                text += ("\n\nGemini is busy or rate-limiting you. Wait a minute and retry, "
                         "try another model, or enable billing in AI Studio.")
            yield {"type": "error", "text": text}
            return
        yield {"type": "answer", "seconds": round(time.time() - started),
               "text": self.link_citations(final_answer) if final_answer
               else "_The agent returned no answer (try raising AGENT_RECURSION_LIMIT)._"}


# ---------------------------------------------------------------------------------------
# 7 · Streamlit UI
# ---------------------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_backend(google_api_key: str, pinecone_api_key: str) -> Backend:
    """Created once per pair of keys and shared by every session and rerun."""
    return Backend(google_api_key, pinecone_api_key)


def require_login():
    """If UI_USERNAME and UI_PASSWORD are set, ask for them before showing the app."""
    username, password = setting("UI_USERNAME"), setting("UI_PASSWORD")
    if not (username and password) or st.session_state.get("logged_in"):
        return
    st.title("🧭 Codebase Onboarding Agent")
    with st.form("login"):
        entered_user = st.text_input("Username")
        entered_password = st.text_input("Password", type="password")
        if st.form_submit_button("Log in", type="primary"):
            if (hmac.compare_digest(entered_user.encode(), username.encode())
                    and hmac.compare_digest(entered_password.encode(), password.encode())):
                st.session_state.logged_in = True
                st.rerun()
            st.error("Wrong username or password.")
    st.stop()


def api_keys():
    """Keys from the environment / secrets file, otherwise from sidebar inputs (kept per session)."""
    google_key, pinecone_key = setting("GOOGLE_API_KEY"), setting("PINECONE_API_KEY")
    if google_key and pinecone_key:
        return google_key, pinecone_key
    with st.sidebar:
        st.subheader("API keys")
        st.caption("Free keys: [Google AI Studio](https://aistudio.google.com/apikey) · "
                   "[Pinecone](https://app.pinecone.io). They stay in this browser session only.")
        if not google_key:
            google_key = st.text_input("GOOGLE_API_KEY", type="password", key="google_key_input").strip()
        if not pinecone_key:
            pinecone_key = st.text_input("PINECONE_API_KEY", type="password", key="pinecone_key_input").strip()
    return google_key, pinecone_key


def render_step(step: dict):
    """One finished agent step (tool call, citation check, notice) from the chat history."""
    if step["kind"] == "tool":
        with st.expander(step["title"]):
            st.code(step.get("result") or "(no result)", language=None)
    elif step["kind"] == "check":
        st.caption(f"🔎 Citation check: {step['text']}")
    else:
        st.caption(step["text"])


def render_message(msg: dict):
    with st.chat_message(msg["role"]):
        for step in msg.get("steps", []):
            render_step(step)
        if msg.get("content"):
            st.markdown(msg["content"])
        if msg.get("seconds") is not None:
            st.caption(f"Answered in {msg['seconds']}s")


def load_repo(backend: Backend, repo_url: str, force_reindex: bool = False):
    """Index a repository, show progress, and start a new conversation about it."""
    if not repo_url.strip():
        st.sidebar.warning("Enter a GitHub repository URL first.")
        return
    bar = st.sidebar.progress(0.0, text="Starting...")
    if backend.lock.locked():
        bar.progress(0.0, text="Another job is running; waiting for my turn...")
    with backend.lock:
        try:
            backend.prepare_repository(repo_url, force_reindex=force_reindex,
                                       progress=lambda frac, text: bar.progress(min(max(frac, 0.0), 1.0), text=text))
        except Exception as e:
            bar.empty()
            st.sidebar.error(f"Could not load the repository: {e}")
            return
        st.session_state.namespace = backend.state["namespace"]
    bar.empty()
    st.session_state.thread_id = str(uuid.uuid4())     # don't carry the previous repo's chat over
    st.session_state.history = []


def answer_question(backend: Backend, question: str):
    """Run the agent on one question, streaming each step into the chat."""
    history = st.session_state.history
    history.append({"role": "user", "content": question})
    render_message(history[-1])

    reply = {"role": "assistant", "content": "", "steps": [], "seconds": None}
    with st.chat_message("assistant"):
        if st.session_state.namespace is None and "github.com" not in question:
            reply["content"] = ("Please load a repository first (sidebar), "
                                "or include a GitHub URL in your question.")
            st.markdown(reply["content"])
            history.append(reply)
            return

        waiting = st.empty()
        if backend.lock.locked():
            waiting.info("⏳ Another question is running; waiting for my turn...")
        with backend.lock:
            waiting.empty()
            backend.activate(st.session_state.namespace)
            open_steps = {}                                  # tool_call id -> (status box, step dict)
            thinking = st.empty()
            thinking.caption("Thinking...")
            try:
                for event in backend.agent_events(question, st.session_state.thread_id):
                    thinking.empty()
                    if event["type"] == "tool_call":
                        args = ", ".join(f"{k}={v!r}" for k, v in event["args"].items())
                        step = {"kind": "tool", "title": f"🔧 {event['name']}({args[:120]})", "result": ""}
                        reply["steps"].append(step)
                        open_steps[event["id"]] = (st.status(step["title"], state="running"), step)
                    elif event["type"] == "tool_result":
                        result = event["content"].strip()
                        result = result[:1500] + ("\n... (truncated)" if len(result) > 1500 else "")
                        box, step = open_steps.pop(event["id"], (None, None))
                        if step is None:
                            step = {"kind": "tool", "title": f"🔧 {event['name']}", "result": ""}
                            reply["steps"].append(step)
                            box = st.status(step["title"], state="running")
                        step["result"] = result
                        box.code(result, language=None)
                        box.update(state="complete")
                    elif event["type"] == "check":
                        reply["steps"].append({"kind": "check", "text": event["text"]})
                        render_step(reply["steps"][-1])
                    elif event["type"] == "slow":
                        reply["steps"].append({"kind": "note", "text":
                                               f"⏱️ That step took {event['seconds']}s; Gemini may be busy."})
                        render_step(reply["steps"][-1])
                    elif event["type"] == "error":
                        reply["content"] = f"❌ {event['text']}"
                        st.error(event["text"])
                    elif event["type"] == "answer":
                        reply["content"], reply["seconds"] = event["text"], event["seconds"]
                        st.markdown(reply["content"])
                        st.caption(f"Answered in {reply['seconds']}s")
                    if event["type"] in ("tool_result", "check"):
                        thinking.caption("Thinking...")
            finally:
                for box, _ in open_steps.values():           # tool calls that never returned
                    box.update(state="error")
                thinking.empty()
                history.append(reply)
                # The agent may have switched repos via load_repository
                st.session_state.namespace = backend.state["namespace"]


def sidebar(backend: Backend):
    with st.sidebar:
        st.subheader("Repository")
        with st.form("load_repo", border=False):
            repo_url = st.text_input("GitHub repository", placeholder="https://github.com/owner/repo")
            load_clicked = st.form_submit_button("Load repository", type="primary", use_container_width=True)
        st.caption("Python repositories only. The first load of a repository can take a few minutes "
                   "while it is embedded; later loads are quick.")

    if load_clicked:
        load_repo(backend, repo_url)

    with st.sidebar:
        repo = backend.repo_cache.get(st.session_state.namespace)
        if repo:
            indexed = repo["summary"].splitlines()[1] if repo.get("summary") else ""
            st.success(f"**[{repo['owner']}/{repo['repo']}]({repo['url']})**\n\n{indexed}")
            left, right = st.columns(2)
            if left.button("🧹 New chat", use_container_width=True, help="Keep the repository, clear the chat"):
                st.session_state.thread_id = str(uuid.uuid4())
                st.session_state.history = []
                st.rerun()
            if right.button("🔄 Re-index", use_container_width=True,
                            help="Fresh clone and rebuild the vectors, e.g. after the repository changed"):
                load_repo(backend, repo["url"], force_reindex=True)
                st.rerun()
        else:
            st.info("No repository loaded.")
        st.caption(f"Model: `{backend.model_name}` · Pinecone index: `{backend.index_name}`")

        with st.expander("Indexed repositories"):
            try:
                namespaces = backend.indexed_namespaces()
            except Exception as e:
                namespaces = {}
                st.caption(f"Could not read the index: {e}")
            if not namespaces:
                st.caption("Nothing indexed yet.")
            for ns, count in sorted(namespaces.items()):
                name, remove = st.columns([4, 1])
                name.markdown(f"`{ns.replace('--', '/', 1)}`  \n{count:,} vectors")
                if remove.button("🗑️", key=f"delete-{ns}", help="Delete this repository's vectors from Pinecone"):
                    with backend.lock:
                        backend.delete_repository_index(ns)
                    if st.session_state.namespace == ns:
                        st.session_state.namespace = None
                        st.session_state.history = []
                    st.rerun()

        graph_png = Path(__file__).parent / "docs" / "agent-graph.png"
        if graph_png.exists():
            with st.expander("How the agent works"):
                st.image(str(graph_png))
                st.caption("agent decides → tools run → check_citations verifies every `file:line` "
                           f"(at most {MAX_TOOL_CALLS} tool calls per question).")


def main():
    st.set_page_config(page_title="Codebase Onboarding Agent", page_icon="🧭", layout="wide")
    require_login()

    st.session_state.setdefault("thread_id", str(uuid.uuid4()))   # conversation id for this visitor
    st.session_state.setdefault("namespace", None)                # this visitor's repository
    st.session_state.setdefault("history", [])                    # chat messages shown on screen

    st.title("🧭 Codebase Onboarding Agent")
    st.caption("Ask questions about any public **Python** GitHub repository. The agent searches the docs "
               "and code, follows the trail, and cites `file:line` for every claim.")

    google_key, pinecone_key = api_keys()
    if not (google_key and pinecone_key):
        st.info("Add your **GOOGLE_API_KEY** and **PINECONE_API_KEY** in the sidebar to get started "
                "(or set them as environment variables before running `streamlit run app.py`).")
        st.stop()

    try:
        with st.spinner("Connecting to Gemini and Pinecone (first run only)..."):
            backend = get_backend(google_key, pinecone_key)
    except Exception as e:
        st.error(f"Could not connect:\n\n{e}")
        st.stop()

    sidebar(backend)

    for msg in st.session_state.history:
        render_message(msg)

    question = None
    if not st.session_state.history:
        st.markdown("**Example questions**")
        for column, example in zip(st.columns(2) + st.columns(2), EXAMPLE_QUESTIONS):
            if column.button(example, use_container_width=True):
                question = example
    typed = st.chat_input("e.g. How does signing work, step by step?")
    question = (typed or question or "").strip()
    if question:
        answer_question(backend, question)


if __name__ == "__main__":
    main()
