# These are downloaded from hugging face hub. Ensure the model name matches
embedding_models = {
    "all-MiniLM-L6-v2": {"name": "sentence-transformers/all-MiniLM-L6-v2", 'kwargs': {'device': 'cpu'}},
    "instructor-base": {"name": "hkunlp/instructor-base", 'kwargs': {"device": "cpu"}},
    "instructor-xl": {"name": "hkunlp/instructor-xl", 'kwargs': {'device': 'cuda'}},
}

# Download these models and place them in your system
llm_models_path = {
    "wizard-vicuna-13B-q5_1": "./models/wizard-vicuna-13B.ggmlv3.q5_1.bin",
    "wizardLM-13B.q5_1": "./models/wizardLM-13B-Uncensored.ggmlv3.q5_1.bin",
    "wizardLM-13B.q4_1": "./models/wizardLM-13B-Uncensored.ggmlv3.q4_1.bin",
}
