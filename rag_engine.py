import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


PERSIST_DIRECTORY = "./chroma_mental_health_kb"

# Internal evidence-based mental health coping knowledge base
KNOWLEDGE_BASE = [
    {
        "topic": "Grounding Exercise (5-4-3-2-1 Technique)",
        "content": "For acute anxiety or panic: Name 5 things you can see, 4 things you can physically feel, 3 things you can hear, 2 things you can smell, and 1 thing you can taste. This shifts activation from the amygdala back to sensory processing."
    },
    {
        "topic": "Box Breathing (Autonomic Regulation)",
        "content": "Inhale slowly for 4 seconds, hold your breath for 4 seconds, exhale smoothly for 4 seconds, and hold empty for 4 seconds. Repeat for 4 cycles to stimulate the vagus nerve and reduce heart rate."
    },
    {
        "topic": "CBT Cognitive Reframing for Exam & Academic Stress",
        "content": "Notice catastrophic cognitive distortions (e.g., 'If I fail, my life is ruined'). Reframe towards evidence: 'This exam is challenging, but it measures my preparation on a single day, not my worth as a person. I will focus on the next actionable step.'"
    },
    {
        "topic": "Sleep Hygiene & Circadian Support",
        "content": "Maintain a regular sleep-wake schedule. Avoid screens 60 minutes before sleeping due to blue light melatonin disruption. Keep the room between 18-20°C and reserve the bed exclusively for sleep."
    },
    {
        "topic": "Progressive Muscle Relaxation (PMR)",
        "content": "Systematically tense muscle groups (forehead, shoulders, fists, calves) for 5 seconds, then release for 15 seconds. Notice the sensation of release to break somatic tension loops."
    }
]

class MentalHealthRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = self._init_store()
        
    def _init_store(self):
        if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
            return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=self.embeddings)
            
        docs = [
            Document(page_content=item["content"], metadata={"topic": item["topic"]})
            for item in KNOWLEDGE_BASE
        ]
        return Chroma.from_documents(docs, self.embeddings, persist_directory=PERSIST_DIRECTORY)
        
    def retrieve_coping_strategies(self, query: str, k: int = 2) -> str:
        results = self.vector_store.similarity_search(query, k=k)
        if not results:
            return ""
        context = "\n---\n".join([f"Strategy ({d.metadata.get('topic', 'General')}): {d.page_content}" for d in results])
        return context