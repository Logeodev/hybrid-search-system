import re
from typing import List
from .document import Document

def preprocess_documents(documents: List[str], chunk_size:int=512) -> List[Document]:
    """Clean and normalize document text
    
    Arguments:
        documents: List of raw document strings
        chunk_size: Maximum number of characters per document text chunk"""
    processed = []
    
    for doc_idx, doc in enumerate(documents):
        # Remove extra whitespace
        doc = re.sub(r'\s+', ' ', doc.strip())
        
        # Handle special characters
        doc = re.sub(r'[^\w\s\.\,\!\?\-]', '', doc)

        doc = doc.lower().strip()  # Normalize case and trim
        
        # Ensure minimum length
        # if len(doc) > 50:  # Skip very short documents
        # Split document into sentences using regex
        sentences = re.findall(r'[^.!?]+[.!?]?', doc)
        chunk = ""
        chunk_num = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # If adding this sentence would exceed chunk_size, save current chunk
            if len(chunk) + len(sentence) > chunk_size and chunk:
                processed.append(
                    Document(idx=doc_idx, text=chunk.strip(), chunk=chunk_num)
                )
                chunk_num += 1
                chunk = ""
            chunk += (sentence + " ")
        # Add any remaining chunk
        if chunk.strip():
            processed.append(Document(idx=doc_idx, text=chunk.strip(), chunk=chunk_num))
    
    return processed
