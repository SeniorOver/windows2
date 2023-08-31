import os

# para usar la libreira de langchain con openia
from langchain.llms import OpenAI

# para trabajar con archivos pdf
from langchain.document_loaders import PyPDFLoader

# Representacion vectorial del texto, toma un texto y lo hace vector ejemplo[0.47358934534895,   0.4234234]
from langchain.embeddings import OpenAIEmbeddings

# almacenamos estos embeddings en chroma
from langchain.vectorstores import Chroma

##Libreria especializada para preguntas y respuestas
from langchain.chains import RetrievalQA

# COMUNICACION CON SERVIDOR STREAMLIT
import streamlit as st

llm = OpenAI(openai_api_key="...")

# PERSONAL
# os.environ['OPENAI_API_KEY'] = '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'

# OTRO
os.environ['OPENAI_API_KEY'] = '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'

default_doc_name = 'doc.pdf'


def process_doc(
        path: str = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf',
        is_local: bool = False,
        question: str = 'Cuál es el titulo del pdf?'
):
    _, loader = os.system(f'curl -o {default_doc_name} {path}'), PyPDFLoader(f"./{default_doc_name}") if not is_local \
        else PyPDFLoader(path)

    doc = loader.load_and_split()

    print(doc[-1])

    ##Si deseamos usar un vector store diferente a Chroma
    ##db = Annoy.from_documents(document, embedding=HuggingFaceEmbeddings())
    db = Chroma.from_documents(doc, embedding=OpenAIEmbeddings())

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type='stuff', retriever=db.as_retriever())

    # st.write(qa.run(question))
    print(qa.run(question))

def client():
    st.title('Manage LLM with LangChain')
    uploader = st.file_uploader('Upload PDF', type='pdf')

    if uploader:
        with open(f'./{default_doc_name}', 'wb') as f:
            f.write(uploader.getbuffer())
        st.success('PDF saved!!')

    question = st.text_input('Generar un resumen de 20 palabras sobre el pdf',
                             placeholder='Give response about your PDF', disabled=not uploader)

    if st.button('Send Question'):
        if uploader:
            process_doc(
                path=default_doc_name,
                is_local=True,
                question=question
            )
        else:
            st.info('Loading default PDF')
            process_doc()

if __name__ == "__main__":
        client()

