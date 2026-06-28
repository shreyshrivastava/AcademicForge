# backend/roadmap_generator.py
def generate_roadmap(papers):
    """Generate an implementation roadmap based on paper summaries"""
    # In a real implementation, this would generate a detailed roadmap
    # For MVP, we'll return a simple roadmap
    return {
        "research_question": "Multilingual fact-checking system",
        "approaches": [
            {"name": "BERT-based system", "status": "In development"},
            {"name": "Cross-lingual system", "status": "Under review"}
        ],
        "next_steps": [
            "Implement BERT-based system",
            "Test cross-lingual capabilities",
            "Evaluate performance metrics"
        ]
    }# backend/prompts.py
def get_search_prompt():
    """Get prompt for paper search"""
    return "Find recent papers on multilingual fact-checking systems"

def get_summarize_prompt():
    """Get prompt for paper summarization"""
    return "Summarize this paper in 150 words"

def get_roadmap_prompt():
    """Get prompt for roadmap generation"""
    return "Create an implementation roadmap for multilingual fact-checking"