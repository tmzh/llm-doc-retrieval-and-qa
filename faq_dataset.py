from langchain.document_loaders import JSONLoader

FILE_PATH = './data/faq_dataset.json'

loader = JSONLoader(
    file_path=FILE_PATH,
    jq_schema='.questions[] | "Question: \(.question) \n Answer: \(.answer)\n"')

dataset = loader.load()
