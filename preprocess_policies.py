#!/usr/bin/env python3
"""
Pre-processing script to parse all policy PDFs and save chunks to disk.
This runs once to avoid the 10-15 minute processing time on every app load.
"""

import os
import pickle
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import io
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Same Chunk class as in your main app
@dataclass
class Chunk:
    text: str
    page: int
    file_name: str

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

def chunk_pages(pages: List[Tuple[int, str]], file_name: str, size: int = 1100, overlap: int = 200) -> List[Chunk]:
    """Chunk pages into text segments."""
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

def preprocess_all_policies():
    """Process all PDFs in the policy folders and save chunks to disk."""
    policy_folders = ["AA", "CMC", "DD", "EE", "FF", "GA", "GG", "HH", "MA", "PA"]
    all_chunks = []
    total_files = 0
    
    print("🔍 Scanning for policy PDFs...")
    
    for folder in policy_folders:
        folder_path = f"samples/{folder}"
        if os.path.exists(folder_path):
            pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
            print(f"📁 Found {len(pdf_files)} PDFs in {folder}/")
            
            for pdf_file in pdf_files:
                pdf_path = os.path.join(folder_path, pdf_file)
                try:
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    
                    pages = read_pdf_bytes(pdf_bytes)
                    chunks = chunk_pages(pages, pdf_file)
                    all_chunks.extend(chunks)
                    total_files += 1
                    
                    print(f"  ✅ Processed {pdf_file}: {len(chunks)} chunks")
                    
                except Exception as e:
                    print(f"  ❌ Error processing {pdf_file}: {e}")
        else:
            print(f"📁 Folder {folder}/ not found")
    
    # Save chunks to disk
    cache_file = "preprocessed_chunks.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump(all_chunks, f)
    
    print(f"\n🎉 Chunk processing complete!")
    print(f"📊 Total files processed: {total_files}")
    print(f"📄 Total chunks created: {len(all_chunks)}")
    print(f"💾 Chunks saved to: {cache_file}")
    print(f"📏 Cache file size: {os.path.getsize(cache_file) / 1024 / 1024:.1f} MB")
    
    # Now create embeddings
    if all_chunks:
        print(f"\n🔄 Creating embeddings for {len(all_chunks)} chunks...")
        
        # Load sentence transformer model
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"📥 Downloading/loading model: {model_name}")
        model = SentenceTransformer(model_name)
        
        # Create embeddings
        texts = [chunk.text for chunk in all_chunks]
        print(f"🧮 Computing embeddings...")
        embeddings = model.encode(texts, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Save embeddings to disk
        embeddings_file = "preprocessed_embeddings.pkl"
        with open(embeddings_file, "wb") as f:
            pickle.dump(embeddings, f)
        
        print(f"✅ Embeddings created and saved!")
        print(f"📐 Embedding shape: {embeddings.shape}")
        print(f"💾 Embeddings saved to: {embeddings_file}")
        print(f"📏 Embeddings file size: {os.path.getsize(embeddings_file) / 1024 / 1024:.1f} MB")
        
        print(f"\n🚀 Everything is now pre-processed!")
        print(f"✨ Your app will load instantly with no processing!")
    else:
        print(f"\n⚠️  No chunks to embed!")

if __name__ == "__main__":
    preprocess_all_policies()