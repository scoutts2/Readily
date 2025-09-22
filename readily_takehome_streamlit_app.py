import os
import io
import json
import time
import math
import asyncio
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, skip
    pass

import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ------------------------------
# Config
# ------------------------------
APP_TITLE = "Readily: Audit Q → Evidence Checker"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1100
CHUNK_OVERLAP = 200
TOP_K = 256                # retrieve this many chunks per question
GROUPS = 16                # number of parallel groups
CHUNKS_PER_GROUP = TOP_K // GROUPS  # 16
FINAL_CONTEXT_K = 16       # best-of-group chunks that feed the final judge
MAX_QUOTES_IN_FINAL = 2
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ------------------------------
# Optional OpenAI (async) client
# ------------------------------
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
if USE_OPENAI:
    try:
        from openai import AsyncOpenAI
        aclient = AsyncOpenAI()
    except Exception as e:
        USE_OPENAI = False

# ------------------------------
# Google Generative AI (Gemini) client (API-key based)
# ------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
USE_GEMINI = bool(os.getenv("GOOGLE_API_KEY"))
if USE_GEMINI:
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    except Exception:
        USE_GEMINI = False

# ------------------------------
# Data classes
# ------------------------------
@dataclass
class Chunk:
    text: str
    page: int
    file_name: str

# ------------------------------
# Utilities
# ------------------------------
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


def read_pdf_bytes(file_bytes: bytes) -> List[Tuple[int, str]]:
    """Return list of (page_number, page_text)."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            pages.append((i, page.extract_text() or ""))
        except Exception:
            pages.append((i, ""))
    return pages


def chunk_pages(pages: List[Tuple[int, str]], file_name: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    chunks: List[Chunk] = []
    for pnum, ptxt in pages:
        t = (ptxt or "").replace("\n", " ").strip()
        if not t:
            continue
        start = 0
        L = len(t)
        while start < L:
            end = min(L, start + size)
            piece = t[start:end].strip()
            if piece:
                chunks.append(Chunk(piece, pnum, file_name))
            if end == L:
                break
            start = max(0, end - overlap)
    return chunks


def extract_questions_from_text(text: str) -> List[str]:
    """Extract questions by splitting text at reference boundaries"""
    import re
    
    # Clean and normalize the text first
    cleaned_text = re.sub(r'\s+', ' ', text)  # Normalize all whitespace
    
    # Step 1: Find all references to know how many questions exist
    refs = re.findall(r'\(Reference[^)]*\)', cleaned_text)
    
    # Step 2: Split text by references to get question chunks
    # Each chunk should contain exactly one question
    reference_splits = re.split(r'(\(Reference[^)]*\))', cleaned_text)
    
    # Reconstruct question+reference pairs
    question_chunks = []
    for i in range(0, len(reference_splits) - 1, 2):
        if i + 1 < len(reference_splits):
            question_part = reference_splits[i]
            reference_part = reference_splits[i + 1]
            if reference_part.startswith('(Reference'):
                question_chunks.append(question_part + ' ' + reference_part)
    
    # Special handling for question 1 - it might be at the very beginning
    if question_chunks and len(question_chunks) < len(refs):
        # Check if there's text before the first reference that contains question 1
        first_ref_pos = cleaned_text.find('(Reference')
        if first_ref_pos > 0:
            potential_q1 = cleaned_text[:first_ref_pos]
            # Look for "1." in this text
            if '1.' in potential_q1:
                q1_start = potential_q1.rfind('1.')
                if q1_start >= 0:
                    q1_text = potential_q1[q1_start:]
                    # Find the reference that follows
                    first_ref_match = re.search(r'\(Reference[^)]*\)', cleaned_text[first_ref_pos:first_ref_pos+200])
                    if first_ref_match:
                        q1_with_ref = q1_text + ' ' + first_ref_match.group(0)
                        question_chunks.insert(0, q1_with_ref)
                        pass  # Question 1 added successfully
    
    # Now we have all question chunks
    
    # Step 3: Process each chunk systematically - assign sequential numbers
    questions = []
    
    for i, chunk in enumerate(question_chunks):
        chunk = chunk.strip()
        if not chunk:
            continue
        
        # Assign sequential number (1, 2, 3, ..., 64)
        expected_number = i + 1
        
        # Process this chunk to extract question text
        
        # Extract the reference first
        ref_match = re.search(r'(\(Reference[^)]*\))$', chunk)
        if ref_match:
            reference = ref_match.group(1)
            question_text = chunk.replace(reference, '').strip()
        else:
            reference = ""
            question_text = chunk
        
        # Aggressive cleaning - remove all unwanted patterns
        question_text = re.sub(r'Yes\s+No\s+Citation:\s*\d+\s*\.?\s*', '', question_text)
        question_text = re.sub(r'Yes\s+No\s+Citation[^?]*?(?=\d+\.)', '', question_text)  # Remove until next number
        question_text = re.sub(r'Yes\s+No\s+Citation:\s*', '', question_text)
        question_text = re.sub(r'Rev\.\s*\d+/\d+', '', question_text)
        
        # Remove any existing numbers at the start (we'll use sequential numbering)
        question_text = re.sub(r'^\d+\.\s*', '', question_text)
        question_text = re.sub(r'\s+', ' ', question_text).strip()
        
        # Extract the complete question content including any text after the ? but before (Reference)
        if 'Does the P&P' in question_text or 'D oes the P&P' in question_text:
            # Find where the actual question starts and capture everything until the reference
            # This includes any explanatory text after the main question
            does_match = re.search(r'(D?\s*oes the P&P.*?)(?=\(Reference)', question_text, re.DOTALL)
            if does_match:
                question_text = does_match.group(1).strip()
            else:
                # Fallback: try the original chunk for complete text
                if '?' in chunk and 'Does the P&P' in chunk:
                    # Find the complete question from the chunk including any text after ?
                    full_match = re.search(r'(D?\s*oes the P&P.*?)(?=\(Reference)', chunk, re.DOTALL)
                    if full_match:
                        question_text = full_match.group(1).strip()
                        # Clean it again
                        question_text = re.sub(r'Yes\s+No\s+Citation:\s*\d+\s*\.?\s*', '', question_text)
                        question_text = re.sub(r'Yes\s+No\s+Citation:\s*', '', question_text)
                        question_text = re.sub(r'Rev\.\s*\d+/\d+', '', question_text)
                        question_text = re.sub(r'^\d+\.\s*', '', question_text)
                        question_text = re.sub(r'\s+', ' ', question_text).strip()
        
        # Final cleanup
        question_text = re.sub(r'\s+', ' ', question_text).strip()
        
        # Be more flexible - accept questions that are substantial even if they don't end with ?
        # (in case the ? got lost in processing)
        substantial_question = len(question_text) > 10 and ('Does the P&P' in question_text or 'D oes the P&P' in question_text)
        ends_with_question = question_text.endswith("?")
        
        if substantial_question:
            # If it doesn't end with ?, try to add it if it should be a question
            if not ends_with_question and ('Does the P&P' in question_text or 'D oes the P&P' in question_text):
                question_text += "?"
            
            full_question = f"{expected_number}. {question_text} {reference}"
            questions.append((expected_number, full_question))
    
    # Step 4: Remove duplicates and sort
    seen_numbers = set()
    unique_questions = []
    
    for num, q in questions:
        if num not in seen_numbers:
            unique_questions.append((num, q))
            seen_numbers.add(num)
    
    # Sort by question number
    unique_questions.sort(key=lambda x: x[0])
    final_questions = [q[1] for q in unique_questions]
    
    return final_questions


def pdf_file_to_text(file_bytes: bytes) -> str:
    pages = read_pdf_bytes(file_bytes)
    return "\n".join(t for _, t in pages)


# ------------------------------
# Embedding + Vector Search
# ------------------------------
@st.cache_resource(show_spinner=False)
def embed_chunks(chunks: List[Chunk]) -> np.ndarray:
    model = load_embedder()
    texts = [c.text for c in chunks]
    if not texts:
        return np.zeros((0,384), dtype=np.float32)
    vecs = model.encode(texts, normalize_embeddings=True)
    return np.array(vecs, dtype=np.float32)


def embed_queries(queries: List[str]) -> np.ndarray:
    model = load_embedder()
    if not queries:
        return np.zeros((0,384), dtype=np.float32)
    vecs = model.encode(queries, normalize_embeddings=True)
    return np.array(vecs, dtype=np.float32)


def topk_indices_for_query(q_vec: np.ndarray, mat: np.ndarray, k: int) -> np.ndarray:
    # cosine since both are normalized → dot product
    sims = mat @ q_vec
    k = min(k, mat.shape[0])
    if k <= 0:
        return np.array([], dtype=int)
    # partial sort is faster than argsort for large arrays
    idx = np.argpartition(-sims, k-1)[:k]
    # order those top-k
    idx = idx[np.argsort(-sims[idx])]
    return idx


# ------------------------------
# LLM Prompts (async)
# ------------------------------
async def llm_pick_best_from_group(question: str, group: List[int], chunks: List[Chunk]) -> Dict[str, Any]:
    """Ask the LLM to pick the single best citation from a group of chunk ids.
    Returns: {"best_global_idx": int, "quote": str, "page": int, "file_name": str}
    If OpenAI not available, return a heuristic pick.
    """
    if not (USE_OPENAI or USE_GEMINI):
        # heuristic: pick longest chunk containing most overlapping words
        q_words = set(question.lower().split())
        best = None
        best_score = -1
        for gi in group:
            c = chunks[gi]
            score = sum(1 for w in q_words if w and w in c.text.lower())
            if score > best_score:
                best_score = score
                best = gi
        gi = best if best is not None else group[0]
        c = chunks[gi]
        return {"best_global_idx": gi, "quote": c.text[:300] + ("..." if len(c.text) > 300 else ""),
                "page": c.page, "file_name": c.file_name}

    # Gemini path (JSON), then OpenAI path
    if USE_GEMINI:
        def fmt_chunk(ci: int) -> str:
            ch = chunks[ci]
            return f"[id={ci} | file={ch.file_name} | page={ch.page}]\n{ch.text}"
        group_text = "\n\n".join(fmt_chunk(ci) for ci in group)
        prompt = (
            "You are a compliance analyst. From the provided policy excerpts, pick the single best citation that answers the question. "
            "Return JSON with keys: id (the provided chunk id), quote (verbatim text), page (int), file_name (string).\n\n"
            f"Question:\n{question}\n\nPolicy excerpts:\n{group_text}"
        )
        try:
            resp = await asyncio.to_thread(
                gemini_model.generate_content,
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            data = json.loads(resp.text)
            out = {
                "best_global_idx": int(data.get("id")),
                "quote": data.get("quote", ""),
                "page": int(data.get("page", 0)),
                "file_name": data.get("file_name", "")
            }
            return out
        except Exception:
            gi = group[0]
            c = chunks[gi]
            return {"best_global_idx": gi, "quote": c.text[:300], "page": c.page, "file_name": c.file_name}

    # Build prompt
    def fmt_chunk(ci: int) -> str:
        ch = chunks[ci]
        return f"[id={ci} | file={ch.file_name} | page={ch.page}]\n{ch.text}"

    group_text = "\n\n".join(fmt_chunk(ci) for ci in group)
    sys = (
        "You are a compliance analyst. From the provided policy excerpts, pick the single best citation that answers the question. "
        "Return JSON with keys: id (the provided chunk id), quote (verbatim text), page (int), file_name (string)."
    )
    user = f"""Question:\n{question}\n\nPolicy excerpts:\n{group_text}\n\nOnly return JSON."""

    try:
        resp = await aclient.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0
        )
        txt = resp.choices[0].message.content.strip()
        data = json.loads(txt)
        out = {
            "best_global_idx": int(data.get("id")),
            "quote": data.get("quote", ""),
            "page": int(data.get("page", 0)),
            "file_name": data.get("file_name", "")
        }
        return out
    except Exception as e:
        # fallback to heuristic
        gi = group[0]
        c = chunks[gi]
        return {"best_global_idx": gi, "quote": c.text[:300], "page": c.page, "file_name": c.file_name}


async def llm_final_judgment(question: str, chosen_ids: List[int], chunks: List[Chunk]) -> Dict[str, Any]:
    """Ask the LLM for Yes/No/Maybe with confidence and up to 2 best citations.
    Returns dict with keys: status, confidence, citations: List[{text, page, file_name}]
    """
    # Build context with FINAL_CONTEXT_K chunks
    chosen_ids = chosen_ids[:FINAL_CONTEXT_K]

    if not (USE_OPENAI or USE_GEMINI):
        # heuristic decision
        # If many overlaps of key words exist in chosen chunks, say Yes; otherwise Maybe
        words = set(w for w in question.lower().split() if len(w) > 3)
        scores = []
        for ci in chosen_ids:
            c = chunks[ci]
            score = sum(1 for w in words if w in c.text.lower())
            scores.append((score, ci))
        scores.sort(reverse=True)
        status = "Yes" if scores and scores[0][0] >= max(3, len(words)//5) else "Maybe"
        conf = max(0.1, min(0.95, 0.3 + 0.1 * (scores[0][0] if scores else 0)))
        cits = []
        for _, ci in scores[:MAX_QUOTES_IN_FINAL]:
            ch = chunks[ci]
            cits.append({"text": ch.text[:400], "page": ch.page, "file_name": ch.file_name})
        return {"status": status, "confidence": round(conf, 2), "citations": cits}

    def fmt(ci: int) -> str:
        ch = chunks[ci]
        return f"[id={ci} | file={ch.file_name} | page={ch.page}]\n{ch.text}"

    context = "\n\n".join(fmt(ci) for ci in chosen_ids)

    # Gemini path (JSON)
    if USE_GEMINI:
        prompt = (
            "You are a careful compliance analyst. Decide if the requirement in the question is satisfied by the policy evidence.\n"
            "Respond with JSON: {status: 'Yes'|'No'|'Maybe', confidence: float in [0,1], citations: [{quote, page, file_name}]}.\n"
            "Citations should be the 1-2 most directly supporting quotes, verbatim, with their page numbers and file names.\n\n"
            f"Question:\n{question}\n\nCandidate evidence excerpts (with ids):\n{context}"
        )
        try:
            resp = await asyncio.to_thread(
                gemini_model.generate_content,
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            data = json.loads(resp.text)
            status = data.get("status", "Maybe")
            if status not in ("Yes","No","Maybe"):
                status = "Maybe"
            conf = float(data.get("confidence", 0.5))
            raw_cits = data.get("citations", [])[:MAX_QUOTES_IN_FINAL]
            norm_cits = []
            for c in raw_cits:
                norm_cits.append({
                    "text": c.get("quote", ""),
                    "page": int(c.get("page", 0)),
                    "file_name": c.get("file_name", "")
                })
            return {"status": status, "confidence": round(max(0.0, min(1.0, conf)), 2), "citations": norm_cits}
        except Exception:
            ch = chunks[chosen_ids[0]] if chosen_ids else None
            cit = ({"text": ch.text[:400], "page": ch.page, "file_name": ch.file_name} if ch else {})
            return {"status": "Maybe", "confidence": 0.5, "citations": [cit] if ch else []}
    sys = (
        "You are a careful compliance analyst. Decide if the requirement in the question is satisfied by the policy evidence.\n"
        "Respond with JSON: {status: 'Yes'|'No'|'Maybe', confidence: float in [0,1], citations: [{quote, page, file_name}]}.\n"
        "Citations should be the 1-2 most directly supporting quotes, verbatim, with their page numbers and file names."
    )
    user = f"""Question:\n{question}\n\nCandidate evidence excerpts (with ids):\n{context}\n\nOnly return JSON."""

    try:
        resp = await aclient.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0
        )
        txt = resp.choices[0].message.content.strip()
        data = json.loads(txt)
        # sanitize
        status = data.get("status", "Maybe")
        if status not in ("Yes","No","Maybe"):
            status = "Maybe"
        conf = float(data.get("confidence", 0.5))
        cits = data.get("citations", [])[:MAX_QUOTES_IN_FINAL]
        # ensure fields exist
        norm_cits = []
        for c in cits:
            norm_cits.append({
                "text": c.get("quote", ""),
                "page": int(c.get("page", 0)),
                "file_name": c.get("file_name", "")
            })
        return {"status": status, "confidence": round(max(0.0, min(1.0, conf)), 2), "citations": norm_cits}
    except Exception:
        # fallback
        ch = chunks[chosen_ids[0]] if chosen_ids else None
        cit = ({"text": ch.text[:400], "page": ch.page, "file_name": ch.file_name} if ch else {})
        return {"status": "Maybe", "confidence": 0.5, "citations": [cit] if ch else []}


# ------------------------------
# Orchestrator per-question
# ------------------------------
async def analyze_question(question: str, chunk_matrix: np.ndarray, chunks: List[Chunk]) -> Dict[str, Any]:
    q_vec = embed_queries([question])[0]
    idxs = topk_indices_for_query(q_vec, chunk_matrix, TOP_K)

    # Group into 16 groups of 16
    groups: List[List[int]] = []
    for g in range(GROUPS):
        start = g * CHUNKS_PER_GROUP
        end = start + CHUNKS_PER_GROUP
        if start < len(idxs):
            groups.append([int(idxs[i]) for i in range(start, min(end, len(idxs)))])

    # parallel: pick best citation per group
    tasks = [llm_pick_best_from_group(question, grp, chunks) for grp in groups]
    group_results = await asyncio.gather(*tasks)

    chosen = [gr["best_global_idx"] for gr in group_results]
    final = await llm_final_judgment(question, chosen, chunks)

    # enrich citations with file/page when missing
    if final.get("citations"):
        for c in final["citations"]:
            if not c.get("file_name") or not c.get("page"):
                # fill from chosen chunk if possible
                ci = chosen[0]
                ch = chunks[ci]
                c.setdefault("file_name", ch.file_name)
                c.setdefault("page", ch.page)

    return {
        "question": question,
        "status": final.get("status", "Maybe"),
        "confidence": final.get("confidence", 0.5),
        "citations": final.get("citations", [])
    }


# ------------------------------
# Persistence (very lightweight)
# ------------------------------
STORAGE_DIR = "storage"
os.makedirs(STORAGE_DIR, exist_ok=True)


def save_results(username: str, results: List[Dict[str, Any]]):
    if not username:
        return
    path = os.path.join(STORAGE_DIR, f"{username}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def load_results(username: str) -> List[Dict[str, Any]]:
    if not username:
        return []
    path = os.path.join(STORAGE_DIR, f"{username}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


# ------------------------------
# UI
# ------------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# Debug: Show API status
google_key = os.getenv("GOOGLE_API_KEY", "")
openai_key = os.getenv("OPENAI_API_KEY", "")

with st.expander("🔧 API Configuration Status", expanded=False):
    if google_key:
        st.success(f"✅ Google API Key: Configured (ends with ...{google_key[-4:]})")
        st.info(f"Using Gemini model: {GEMINI_MODEL}")
    else:
        st.warning("⚠️ Google API Key: Not configured")
        
    if openai_key:
        st.success(f"✅ OpenAI API Key: Configured (ends with ...{openai_key[-4:]})")
    else:
        st.warning("⚠️ OpenAI API Key: Not configured")
        
    if not google_key and not openai_key:
        st.error("❌ No AI API keys configured - using basic heuristic (90% confidence cap)")
        st.info("💡 Set GOOGLE_API_KEY or OPENAI_API_KEY environment variable for better analysis")

# Enhanced sign-in system
with st.sidebar:
    # Check if user is already logged in
    if "logged_in_user" in st.session_state and st.session_state["logged_in_user"]:
        st.subheader("Account")
        st.success(f"Logged in as **{st.session_state['logged_in_user']}**")
        if st.button("Sign Out"):
            st.session_state["logged_in_user"] = None
            st.session_state["results"] = []
            st.rerun()
        
        # Show saved results for logged-in user
        saved_results = load_results(st.session_state["logged_in_user"])
        if saved_results:
            st.caption(f"📊 {len(saved_results)} saved results")
            if st.button("Load Saved Results"):
                st.session_state["results"] = saved_results
                st.success("Loaded saved results!")
                st.rerun()
    else:
        st.subheader("Account")
        st.info("👤 **Visiting as Guest**")
        st.caption("Sign in for persistent results")
        
        with st.form("signin_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign In", type="primary")
            
            if submitted:
                if username and password:
                    # Simple validation (in real app, this would check against a database)
                    if username == "admin" and password == "admin":
                        st.session_state["logged_in_user"] = username
                        st.success(f"Welcome back, {username}!")
                        st.rerun()
                    elif username == "readily_employee" and password == "readily":
                        st.session_state["logged_in_user"] = username
                        st.success(f"Welcome back, {username}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.error("Please enter both username and password")
        
        # Create Account Section
        st.markdown("---")
        st.caption("Don't have an account?")
        
        with st.form("create_account_form"):
            new_username = st.text_input("Choose Username", placeholder="Enter new username", key="new_username")
            new_password = st.text_input("Choose Password", type="password", placeholder="Enter new password", key="new_password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm password", key="confirm_password")
            create_submitted = st.form_submit_button("Create Account", type="secondary")
            
            if create_submitted:
                if new_username and new_password and confirm_password:
                    if new_password == confirm_password:
                        if len(new_username) >= 3 and len(new_password) >= 6:
                            # Check if username already exists (in real app, check database)
                            existing_users = ["admin", "readily_employee"]
                            if new_username not in existing_users:
                                # In a real app, you would save to database here
                                st.session_state["logged_in_user"] = new_username
                                st.success(f"Account created successfully! Welcome, {new_username}!")
                                st.rerun()
                            else:
                                st.error("Username already exists. Please choose a different username.")
                        else:
                            st.error("Username must be at least 3 characters and password at least 6 characters.")
                    else:
                        st.error("Passwords do not match.")
                else:
                    st.error("Please fill in all fields.")

st.markdown("---")

# Uploads / Auto-upload
colA, colB = st.columns(2)
with colA:
    st.subheader("Policies")
    st.info("✅ Sample policies are pre-loaded and ready!")
    
    pol_files = st.file_uploader("Drop additional policy PDF(s)", type=["pdf"], accept_multiple_files=True, key="pols")
    if pol_files:
        st.success(f"📄 Added {len(pol_files)} additional policy file(s)")

with colB:
    st.subheader("Questions")
    q_file = st.file_uploader("Drop questions PDF", type=["pdf"], accept_multiple_files=False, key="qs")
    if st.button("Auto-load sample questions"):
        sample_q = "samples/questions.pdf"
        if os.path.exists(sample_q):
            with open(sample_q, "rb") as f:
                st.session_state["sample_questions_bytes"] = f.read()
            st.success("Loaded sample questions from samples/questions.pdf")
        else:
            st.warning("samples/questions.pdf not found in repo.")

# Load pre-processed chunks from cache
chunks: List[Chunk] = []
if os.path.exists("preprocessed_chunks.pkl"):
    # Discrete chunk loading
    with st.spinner("Loading policy chunks..."):
        import pickle
        with open("preprocessed_chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
    
    st.caption(f"✅ Loaded {len(chunks)} pre-processed policy chunks")

# Process any additional uploaded files
policy_bytes_list: List[bytes] = []
if pol_files:
    for pf in pol_files:
        policy_bytes_list.append(pf.read())
    # Process additional files and add to existing chunks
    with st.spinner("Processing additional policy files..."):
        for i, b in enumerate(policy_bytes_list):
            pages = read_pdf_bytes(b)
            file_name = getattr(pol_files[i], "name", f"additional_policy_{i+1}.pdf")
            additional_chunks = chunk_pages(pages, file_name)
            chunks.extend(additional_chunks)
        st.success(f"✅ Added {len(additional_chunks)} chunks from additional files!")

questions_bytes: bytes = None
if q_file:
    questions_bytes = q_file.read()
if st.session_state.get("sample_questions_bytes") and not questions_bytes:
    questions_bytes = st.session_state["sample_questions_bytes"]

# Extract questions
if questions_bytes:
    with st.spinner("Parsing questions PDF..."):
        q_text = pdf_file_to_text(questions_bytes)
        questions = extract_questions_from_text(q_text)
        
        # Simple debug
        st.caption(f"Extracted {len(questions)} questions from PDF")
else:
    questions = []

# Chunks are now loaded from cache above

# Load pre-processed embeddings or create new ones
if chunks:
    if os.path.exists("preprocessed_embeddings.pkl"):
        # Discrete loading section
        with st.spinner("Loading embeddings..."):
            import pickle
            with open("preprocessed_embeddings.pkl", "rb") as f:
                chunk_matrix = pickle.load(f)
        
        st.caption(f"✅ Loaded pre-processed embeddings for {len(chunks)} policy chunks")
    else:
        # Discrete embedding creation section
        with st.spinner("Creating embeddings..."):
            chunk_matrix = embed_chunks(chunks)
        
        st.caption(f"✅ Created embeddings for {len(chunks)} chunks")
else:
    chunk_matrix = np.zeros((0,384), dtype=np.float32)

# Allow user to add ad-hoc question
st.markdown("### Add an additional question")
new_q = st.text_input("Type a question and press Enter", value="")
if new_q.strip():
    questions.append(new_q.strip())

# Show message if no questions found
if not questions:
    st.info("Upload or auto-load a questions PDF to begin analysis.")

# Store & show questions
st.markdown("---")
st.subheader("Questions")
if not questions:
    st.info("Upload or auto-load a questions PDF. Questions ending with '?' will appear here.")
else:
    pass  # Questions found, continue
    
    # Results state
    st.session_state.setdefault("results", [])
    
    # Processing options
    st.subheader("Processing Options")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Process All Questions", type="primary"):
            # Create a prominent progress container
            progress_container = st.container()
            status_container = st.container()
            
            with progress_container:
                st.subheader("🔄 Processing All Questions")
                overall_progress = st.progress(0)
                overall_status = st.empty()
            
            # Process all questions at once with visible progress
            all_results = []
            for i, question in enumerate(questions):
                # Update overall progress
                progress_pct = (i + 1) / len(questions)
                overall_progress.progress(progress_pct)
                overall_status.text(f"Processing question {i+1} of {len(questions)}...")
                
                # Show individual question progress
                with status_container:
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.write(f"**Q{i+1}:** {question}")
                    with col_b:
                        question_progress = st.progress(0)
                
                # Process the question with animated progress
                question_progress.progress(0.3)
                result = asyncio.run(analyze_question(question, chunk_matrix, chunks))
                question_progress.progress(1.0)
                
                all_results.append(result)
                
                # Small delay for visual effect
                import time
                time.sleep(0.2)
            
            # Clear progress indicators
            progress_container.empty()
            status_container.empty()
            
            st.session_state["results"] = all_results
            st.success(f"✅ Successfully processed {len(all_results)} questions!")
            st.rerun()
    
    # Second column - could add other features here in the future
    with col2:
        st.write("")  # Empty for now

    # Display questions with pagination (10 per page)
    st.subheader("All Questions")
    
    # Pagination settings
    page_size = 10
    total_pages = (len(questions) + page_size - 1) // page_size
    current_page = st.session_state.get("questions_page", 1)
    
    # Simple pagination controls
    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("⬅️ Previous", disabled=(current_page == 1)):
                st.session_state["questions_page"] = current_page - 1
                st.rerun()
        
        with col2:
            st.markdown(f"**Page {current_page} of {total_pages}**")
        
        with col3:
            if st.button("Next ➡️", disabled=(current_page == total_pages)):
                st.session_state["questions_page"] = current_page + 1
                st.rerun()
        
        st.markdown("---")
    
    # Display questions for current page
    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, len(questions))
    
    for i in range(start_idx, end_idx):
        q = questions[i]
        with st.expander(f"Q{i+1}"):
            st.text(q)
    
    # Show page info at bottom
    if total_pages > 1:
        st.caption(f"Showing questions {start_idx + 1}-{end_idx} of {len(questions)}")


# ------------------------------
# Analysis table + Sort/Group + Export
# ------------------------------
st.markdown("---")
st.subheader("Results")
results: List[Dict[str, Any]] = st.session_state.get("results", [])
clean_results = [r for r in results if r]

if clean_results:
    # Display questions word-for-word
    st.subheader("Questions from PDF:")
    for i, result in enumerate(clean_results):
        question = result.get("question", "")
        st.write(f"**Q{i+1}:**")
        st.markdown(f"```\n{question}\n```")
    
    st.markdown("---")
    
    # sorting/grouping
    sort_by = st.selectbox("Sort by", ["None","Status","Confidence"], index=1)
    if sort_by == "Status":
        order = {"Yes":0, "Maybe":1, "No":2}
        clean_results = sorted(clean_results, key=lambda r: order.get(r.get("status","Maybe"), 1))
    elif sort_by == "Confidence":
        clean_results = sorted(clean_results, key=lambda r: r.get("confidence", 0.0), reverse=True)

    # show table
    def row_of(r: Dict[str,Any]) -> Dict[str, Any]:
        c1 = r.get("citations", [{}])
        c2 = r.get("citations", [{},{}])
        c1 = c1[0] if len(c1) > 0 else {}
        c2 = c2[1] if len(c2) > 1 else {}
        return {
            "Question": r.get("question",""),
            "Status": r.get("status",""),
            "Confidence": r.get("confidence",0.0),
            "Citation1": f"{c1.get('file_name','')} p.{c1.get('page','')}",
            "Citation1 text": c1.get("text","")[:180],
            "Citation2": f"{c2.get('file_name','')} p.{c2.get('page','')}",
            "Citation2 text": c2.get("text","")[:180],
        }

    # Display results in a more readable format
    for i, result in enumerate(clean_results):
        with st.expander(f"Question {i+1} - {result.get('status', 'Unknown')} (Confidence: {result.get('confidence', 0.0):.2f})"):
            # Display question word-for-word
            st.write("**Question:**")
            st.markdown(f"```\n{result.get('question', '')}\n```")
            
            # Display citations
            citations = result.get("citations", [])
            if citations:
                st.write("**Citations:**")
                for j, citation in enumerate(citations[:2]):  # Show top 2 citations
                    if citation:
                        st.write(f"**Citation {j+1}:** {citation.get('file_name', '')} (Page {citation.get('page', '')})")
                        st.write(f"*Text:* {citation.get('text', '')[:200]}...")
    
    # Condensed Results Table
    st.subheader("📋 Results Summary")
    
    # Create condensed table data
    condensed_data = []
    for i, result in enumerate(clean_results):
        status = result.get("status", "Unknown")
        confidence = result.get("confidence", 0.0)
        
        # Create status with confidence bar
        confidence_pct = int(confidence * 100)
        status_display = f"{status} ({confidence_pct}%)"
        
        # Extract the original question number from the question text
        question_text = result.get("question", "")
        question_number = "?"
        if question_text:
            # Look for pattern like "1. " or "2. " at the start
            import re
            match = re.match(r'^(\d+)\.\s*', question_text)
            if match:
                question_number = match.group(1)
        
        condensed_data.append({
            "Question #": f"Q{question_number}",
            "Status": status_display,
            "Confidence": confidence_pct
        })
    
    # Display condensed table
    import pandas as pd
    df_condensed = pd.DataFrame(condensed_data)
    
    # Custom styling for the table
    def style_status(val):
        if "Yes" in val:
            return "background-color: #d4edda; color: #155724"
        elif "Maybe" in val:
            return "background-color: #fff3cd; color: #856404"
        elif "No" in val:
            return "background-color: #f8d7da; color: #721c24"
        return ""
    
    styled_df = df_condensed.style.applymap(style_status, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Add hover information using expandable sections
    st.caption("💡 Click on question numbers below for full details:")
    
    # Create expandable sections for each result with full details and individual buttons
    for i, result in enumerate(clean_results):
        status = result.get("status", "Unknown")
        confidence = result.get("confidence", 0.0)
        question = result.get("question", "")
        
        # Extract the original question number from the question text
        question_number = "?"
        if question:
            import re
            match = re.match(r'^(\d+)\.\s*', question)
            if match:
                question_number = match.group(1)
        
        with st.expander(f"Q{question_number}: {status} ({int(confidence*100)}%)", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("**Full Question:**")
                st.text(question)
                
                # Show citations
                citations = result.get("citations", [])
                if citations:
                    st.write("**Citations:**")
                    for j, citation in enumerate(citations[:2]):
                        if citation:
                            st.write(f"**Citation {j+1}:** {citation.get('file_name', '')} (Page {citation.get('page', '')})")
                            st.write(f"*Text:* {citation.get('text', '')[:300]}...")
            
            with col2:
                # Show Process or Clear button based on whether result exists
                if result and result.get("status") != "Unknown":
                    if st.button("🗑️ Clear", key=f"clear_{i}"):
                        # Remove this result from the results list
                        if "results" in st.session_state and i < len(st.session_state["results"]):
                            st.session_state["results"][i] = None
                        st.rerun()
                else:
                    if st.button("⚡ Process", key=f"process_{i}"):
                        # Process this individual question
                        with st.spinner(f"Processing question {i+1}..."):
                            individual_result = asyncio.run(analyze_question(question, chunk_matrix, chunks))
                            # Update the results list
                            if "results" not in st.session_state:
                                st.session_state["results"] = []
                            while len(st.session_state["results"]) <= i:
                                st.session_state["results"].append(None)
                            st.session_state["results"][i] = individual_result
                        st.rerun()
    
    # Keep original table for export purposes (hidden)
    table = [row_of(r) for r in clean_results]

    # Export CSV at the top
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("📊 Download CSV", type="primary"):
            import pandas as pd
            df = pd.DataFrame(table)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Save results.csv", data=csv_bytes, file_name="results.csv", mime="text/csv")

    # save persistence
    if st.session_state.get("logged_in_user"):
        if st.button("Save results to account"):
            save_results(st.session_state["logged_in_user"], clean_results)
            st.success("Saved.")
else:
    st.caption("No results yet. Process questions to see results here.")

st.markdown("---")
st.caption("Tip: tune CHUNK_SIZE / TOP_K / GROUPS. Set OPENAI_API_KEY for OpenAI **or** GOOGLE_API_KEY for Gemini (Google AI) to enable LLM analysis. You can also set GEMINI_MODEL or OPENAI_MODEL.")
