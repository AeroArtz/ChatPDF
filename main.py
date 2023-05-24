import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader


import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

def load_doc(pdf_doc):

    loader = PyMuPDFLoader(pdf_doc.name)
    documents = loader.load()
    embedding = HuggingFaceEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text = text_splitter.split_documents(documents)
    db = Chroma.from_documents(text, embedding)
    llm = HuggingFaceHub(repo_id="OpenAssistant/oasst-sft-1-pythia-12b", model_kwargs={"temperature": 1.0, "max_length": 256})
    global chain
    chain = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=db.as_retriever())
    return 'Document has successfully been loaded'

def answer_query(query):
    question = query
    return chain.run(question)
html = """
<div style="text-align:center; max width: 700px;">
    <h1>ChatPDF</h1>
    <p> Upload a PDF File, then click on Load PDF File <br>
    Once the document has been loaded you can begin chatting with the PDF =)
</div>"""
css = """container{max-width:700px; margin-left:auto; margin-right:auto,padding:20px}"""
with gr.Blocks(css=css,theme=gr.themes.Monochrome()) as demo:
    gr.HTML(html)
    with gr.Column():
        gr.Markdown('ChatPDF')
        pdf_doc = gr.File(label="Load a pdf",file_types=['.pdf','.docx'],type='file')
        with gr.Row():
            load_pdf = gr.Button('Load pdf file')
            status = gr.Textbox(label="Status",placeholder='',interactive=False)


        with gr.Row():
            input = gr.Textbox(label="type in your question")
            output = gr.Textbox(label="output")
        submit_query = gr.Button("submit")

        load_pdf.click(load_doc,inputs=pdf_doc,outputs=status)

        submit_query.click(answer_query,input,output)

demo.launch()