from langchain import LlamaCpp

from config import llm_models_path

n_gpu_layers = 41  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
model_name = "wizardLM-13B.q5_1"

llm = LlamaCpp(
    model_path=llm_models_path[model_name],
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=None,
    verbose=True,
    n_ctx=2048,
    temperature=0
)
