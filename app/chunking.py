from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
    
def get_code_splitter(file_extension, chunk_size=1000, chunk_overlap=100):
    extension_map = {
        '.py': Language.PYTHON,
        '.js': Language.JS,
        '.ts': Language.TS,
        '.go': Language.GO,
        '.cpp': Language.CPP,
        '.java': Language.JAVA,
        '.php': Language.PHP,
        '.rb': Language.RUBY,
        '.rs': Language.RUST,
    }
    
    lang = extension_map.get(file_extension.lower())
    if lang:
        return RecursiveCharacterTextSplitter.from_language(
            language=lang, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
    else:
        RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )