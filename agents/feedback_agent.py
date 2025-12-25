# agents/feedback_agent.py
def needs_refinement(answer):
    return "I don't know" in answer or len(answer) < 100
