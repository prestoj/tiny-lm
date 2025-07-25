"""Text cleaning utilities for Gutenberg texts"""

import re
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def clean_gutenberg_text(text: str) -> str:
    """Remove Project Gutenberg headers/footers and clean text"""
    # Remove everything before the start marker
    start_patterns = [
        r'\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*',
        r'\*\*\*\s*START OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*',
        # Handle cases with different formatting
        r'\*\*\*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*',
        r'\*\*\*START OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*'
    ]
    
    for pattern in start_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            text = text[match.end():]
            break
    
    # Remove everything after the end marker
    end_patterns = [
        r'\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*',
        r'\*\*\*\s*END OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*',
        # Handle cases with different formatting
        r'\*\*\*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*',
        r'\*\*\*END OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*'
    ]
    
    for pattern in end_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            text = text[:match.start()]
            break
    
    # Also remove "End of Project Gutenberg" lines that appear at the end
    # These sometimes appear before the *** END marker
    end_line_patterns = [
        r'\n*End of (?:the )?Project Gutenberg.*?(?:\n|$)',
        r'\n+\*+END (?:THE|OF) (?:SMALL PRINT|PROJECT GUTENBERG).*?(?:\n|$)'
    ]
    
    for pattern in end_line_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()


def load_and_clean_books(
    books_dir: str,
    metadata_path: str,
    max_books: int = None,
    min_length: int = 1000
):
    """Load and clean all books into memory"""
    
    print(f"Loading metadata from {metadata_path}")
    df = pd.read_csv(metadata_path)
    
    # Get English books
    english_books = df[df['Language'] == 'en']['Etext Number'].astype(str).tolist()
    print(f"Found {len(english_books):,} English books in metadata")
    
    # Filter to existing files
    books_path = Path(books_dir)
    existing_books = []
    for book_id in english_books:
        book_file = books_path / book_id
        if book_file.exists():
            existing_books.append(str(book_file))
    
    print(f"Found {len(existing_books):,} English books on disk")
    
    if max_books:
        existing_books = existing_books[:max_books]
        print(f"Limited to {max_books} books")
    
    # Load all books into memory
    print(f"\nLoading {len(existing_books)} books into memory...")
    cleaned_texts = []
    total_chars = 0
    skipped = 0
    
    for book_path in tqdm(existing_books, desc="Loading books"):
        try:
            with open(book_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()
            
            # Clean text and lowercase
            cleaned_text = clean_gutenberg_text(raw_text).lower()
            
            if len(cleaned_text) < min_length:
                skipped += 1
                continue
            
            cleaned_texts.append(cleaned_text)
            total_chars += len(cleaned_text)
            
        except Exception:
            skipped += 1
            continue
    
    print(f"\nLoaded {len(cleaned_texts)} books into memory")
    print(f"Skipped {skipped} books (too short or errors)")
    print(f"Total characters: {total_chars:,} ({total_chars / 1024 / 1024:.1f} MB)")
    
    return cleaned_texts