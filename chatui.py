from langchain.chains import RetrievalQAWithSourcesChain

from embeddings import get_retriever
from faq_dataset import dataset
import gradio as gr

from llama_cpp import llm

retriever = get_retriever(dataset)

chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True)

css = """ 
    #output_container_0 div.eta-bar {
    display: none !important; transform: none !important;
    }
"""


def format_answer(answer_dict):
    sources = [(doc.page_content.split("\n")[0].replace("Question: ", "").strip(),
                doc.page_content.split("\n")[1].replace("Answer: ", "").strip())
               for doc in answer_dict["source_documents"]]
    answer = answer_dict["answer"]
    references = "## References\n" + "\n\n".join(f"**{q}**\n\n > {a}" for q, a in sources)
    return answer, references


def generate_response(query):
    generated_text = chain(query)
    answer, references = format_answer(generated_text)
    return {answer_block: answer, references_block: references}


with gr.Blocks(css=css, theme=gr.themes.Monochrome()) as faq_bot:
    gr.Markdown("Talk to our FAQ bot")

    with gr.Row():
        with gr.Column():
            answer_block = gr.Textbox(label="Answers", lines=2)
        with gr.Column():
            references_block = gr.Markdown("## References")
    inputs = gr.Textbox(label="Type your question here")

    with gr.Row():
        submit_btn = gr.Button("Ask")
        clear_btn = gr.ClearButton([inputs, answer_block, references_block])

    submit_btn.click(fn=generate_response,
                     inputs=inputs,
                     outputs=[answer_block, references_block],
                     show_progress=False)
    examples_block = gr.Examples(
        ["How can I create an account?",
         "What is the return policy?",
         "How can I contact customer support?"], inputs)

faq_bot.launch()
