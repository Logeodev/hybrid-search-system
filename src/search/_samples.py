from documents import Document

_raw_documents = [
    "Machine learning algorithms require large datasets for training.",
    "Deep learning models use neural networks with multiple layers.",
    "Natural language processing enables computers to understand text.",
    "Computer vision systems can identify objects in images.",
    "Reinforcement learning agents learn through trial and error.",
    "Supervised learning uses labeled data to train models."
]

documents = [Document(idx=i, text=text) for i, text in enumerate(_raw_documents)]

queries = [
    "What are the requirements for machine learning algorithms?",
    "How do deep learning models work?",
    "What is natural language processing?"
]

ground_truth = [
    {0, 5, 4},
    {1, 4},
    {2}
]