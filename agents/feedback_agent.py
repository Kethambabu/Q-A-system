# agents/feedback_agent.py
def needs_refinement(answer):
    """
    Determine if an answer needs refinement.
    Returns True if the answer is incomplete or indicates uncertainty.
    """
    if not answer or not answer.strip():
        return True
    
    # Check for uncertainty markers
    uncertainty_markers = [
        "I don't know",
        "I'm not sure",
        "unclear",
        "no information",
        "unable to"
    ]
    
    answer_lower = answer.lower()
    has_uncertainty = any(marker.lower() in answer_lower for marker in uncertainty_markers)
    
    # Refinement should be triggered for genuinely short answers that aren't the standard "not available" message
    is_too_short = len(answer) < 50 and "is not available in the provided document" not in answer
    
    return has_uncertainty or is_too_short
