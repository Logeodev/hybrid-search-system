import re
from typing import List

def preprocess_documents(documents: List[str]) -> List[str]:
    """Clean and normalize document text"""
    processed = []
    
    for doc in documents:
        # Remove extra whitespace
        doc = re.sub(r'\s+', ' ', doc.strip())
        
        # Handle special characters
        doc = re.sub(r'[^\w\s\.\,\!\?\-]', '', doc)

        # Final strip
        doc = doc.strip()
        
        # Ensure minimum length
        if len(doc) > 50:  # Skip very short documents
            processed.append(doc)
    
    return processed
