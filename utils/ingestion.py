import fitz
from bs4 import BeautifulSoup
import re
from typing import List, Union


def _is_table_like(text: str) -> bool:
    """
    Detect if a text block looks like a table based on structural cues.
    Tables typically have:
    - Multiple numbers/percentages on the same line
    - Tab-separated or multi-space-separated columns
    - Repeated row patterns with numbers
    """
    lines = text.strip().split('\n')
    if len(lines) < 2:
        return False

    # Count lines that contain numbers/percentages
    numeric_lines = 0
    multi_column_lines = 0
    for line in lines:
        # Check for numbers/percentages/decimals
        if re.search(r'\d+\.?\d*\s*%?', line):
            numeric_lines += 1
        # Check for tab-separated or multi-space-separated content (table columns)
        if re.search(r'\t', line) or re.search(r'  {2,}', line):
            multi_column_lines += 1

    # Heuristic: if ≥50% lines have numbers AND ≥30% have column separators → table
    total = len(lines)
    if total >= 2 and numeric_lines / total >= 0.4 and multi_column_lines / total >= 0.25:
        return True

    # Also detect pipe-separated tables: | col1 | col2 |
    pipe_lines = sum(1 for line in lines if line.count('|') >= 2)
    if pipe_lines / total >= 0.5:
        return True

    return False


def _contains_numerical_data(text: str) -> bool:
    """Check if text contains meaningful numerical data (not just page numbers)."""
    # Match patterns like: 94.5%, 0.92, 85.3, etc.
    numeric_matches = re.findall(r'\d+\.\d+%?|\d+%', text)
    # Filter out likely page numbers (single small integers)
    meaningful = [m for m in numeric_matches if '.' in m or '%' in m or len(m) > 2]
    return len(meaningful) >= 2  # At least 2 meaningful numbers


def clean_text(text: str, preserve_structure: bool = False) -> str:
    """
    Normalize whitespace and remove PDF noise.
    If preserve_structure=True, keeps table alignment intact.
    """
    # Fix words broken by hyphens across lines: "inter-\nnational" -> "international"
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    if preserve_structure:
        # For tables: normalize each line individually, keep line breaks
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Collapse only horizontal whitespace runs (not tabs for table alignment)
            line = re.sub(r'[ ]{3,}', '  |  ', line)  # Replace wide gaps with pipe separators
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)
    else:
        # Standard cleaning: collapse all whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


def ingest_pdf_bytes(pdf_bytes: bytes) -> List[str]:
    """
    Ingest PDF using PyMuPDF (fitz) block-level extraction.
    Outputs a list of clean text segments optimized for Tree Search (100–2000 chars).

    CRITICAL FIX: Detects table-like blocks and preserves their structure
    (column alignment, numbers, percentages) instead of collapsing whitespace.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    segments = []
    current_segment = ""
    current_is_table = False
    MAX_SEG_SIZE = 2000
    MIN_SEG_SIZE = 100

    for page in doc:
        # block contains tuples: (x0, y0, x1, y1, text, block_no, block_type)
        blocks = page.get_text("blocks")

        for b in blocks:
            # Type 0 = text block; Type 1 = image block
            if b[6] != 0:
                continue

            raw_block_text = b[4]

            # Detect if this block is table-like BEFORE cleaning
            block_is_table = _is_table_like(raw_block_text)
            block_has_numbers = _contains_numerical_data(raw_block_text)

            # Apply appropriate cleaning strategy
            if block_is_table or block_has_numbers:
                block_text = clean_text(raw_block_text, preserve_structure=True)
                # Tag table blocks so downstream agents know
                if block_is_table:
                    block_text = "[TABLE_DATA]\n" + block_text + "\n[/TABLE_DATA]"
                elif block_has_numbers:
                    block_text = "[NUMERICAL_DATA]\n" + block_text + "\n[/NUMERICAL_DATA]"
            else:
                block_text = clean_text(raw_block_text, preserve_structure=False)

            if not block_text:
                continue

            # If the current segment is a different type (table vs text), flush first
            if current_segment and (block_is_table != current_is_table):
                if len(current_segment) >= MIN_SEG_SIZE:
                    segments.append(current_segment.strip())
                    current_segment = block_text
                    current_is_table = block_is_table
                else:
                    current_segment += "\n\n" + block_text
                continue

            current_is_table = block_is_table

            # Check segment size limits
            if len(current_segment) + len(block_text) > MAX_SEG_SIZE:
                if len(current_segment) >= MIN_SEG_SIZE:
                    segments.append(current_segment.strip())
                    current_segment = block_text
                else:
                    current_segment += "\n\n" + block_text
                    segments.append(current_segment.strip())
                    current_segment = ""
            else:
                current_segment += "\n\n" + block_text if current_segment else block_text

    # Flush remaining segment
    if current_segment.strip():
        if len(current_segment.strip()) >= MIN_SEG_SIZE or not segments:
            segments.append(current_segment.strip())
        elif segments:
            segments[-1] += "\n\n" + current_segment.strip()

    return segments


def ingest_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ")
