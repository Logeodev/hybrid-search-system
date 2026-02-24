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
        for offset in range(0, len(doc), chunk_size):
            chunk = doc[offset:min(offset+chunk_size, len(doc))]
            if chunk:
                processed.append(Document(idx=doc_idx, text=chunk, chunk=offset//chunk_size))
    
    return processed
