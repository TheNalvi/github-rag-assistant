def chunk_document(text, chunk_size=500, chunk_overlap=50):
    chunks = []
    text_len = len(text)
    start = 0
    
    while start < text_len:
        chunk = text[start:start + chunk_size - chunk_overlap]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
        
    return chunks