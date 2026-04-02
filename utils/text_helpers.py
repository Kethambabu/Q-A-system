import re

MAX_SEGMENT_CHARS = 2000
MIN_SEGMENT_CHARS = 200

def split_into_segments(text: str) -> list:
    """
    Split document text into logical segments by paragraph boundaries.
    """
    raw_paragraphs = re.split(r'\n\s*\n', text)
    segments = []
    current_segment = ""
    
    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue
        
        if current_segment and (len(current_segment) + len(para) + 2) > MAX_SEGMENT_CHARS:
            segments.append(current_segment.strip())
            current_segment = para
        else:
            current_segment = (current_segment + "\n\n" + para).strip() if current_segment else para
            
    if current_segment.strip():
        segments.append(current_segment.strip())
        
    if len(segments) <= 1 and len(text) > MAX_SEGMENT_CHARS:
        segments = split_by_sentences(text)
        
    segments = merge_tiny_segments(segments)
    return segments

def split_by_sentences(text: str) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    segments = []
    current = ""
    for sent in sentences:
        if current and (len(current) + len(sent) + 1) > MAX_SEGMENT_CHARS:
            segments.append(current.strip())
            current = sent
        else:
            current = (current + " " + sent).strip() if current else sent
    if current.strip():
        segments.append(current.strip())
    return segments

def merge_tiny_segments(segments: list) -> list:
    if len(segments) <= 1:
        return segments
    merged = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        while i + 1 < len(segments) and len(seg) < MIN_SEGMENT_CHARS:
            i += 1
            seg = seg + "\n\n" + segments[i]
        merged.append(seg)
        i += 1
    return merged
