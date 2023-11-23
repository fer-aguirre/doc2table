import os
import click
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import OpenAI
import pandas as pd


@click.command()
@click.argument('directory_path', type=click.Path(exists=True))
@click.argument('file_path', type=click.Path(exists=True))
@click.argument('token', type=click.STRING)
def extract_information(directory_path, file_path, token):
    pdfs, filenames = load_pdfs(directory_path)
    questions = load_questions(file_path)
    token = save_token(token)

    MAX_TOKENS = 512  # Maximum number of tokens allowed by the model
    # Calculate chunk size and overlap based on maximum tokens
    chunk_size = MAX_TOKENS - 100  # Subtracting a buffer of tokens for safety
    chunk_overlap = chunk_size // 10  # Using 10% overlap
    docs = []
    for pdf in pdfs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                        chunk_overlap=chunk_overlap)
        doc_splits = text_splitter.split_documents(pdf)
        docs.append(doc_splits)

    results = []
    for doc_n, doc in enumerate(docs):
        embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
        # docsearch = Chroma.from_documents(doc, embeddings)
        for i in range(len(doc)):
            doc[i].metadata['source'] = f"snippet-{i}"

        docsearch = Chroma.from_documents(doc, embeddings)

        openai_api_key = token

        chain = RetrievalQAWithSourcesChain.from_chain_type(
                OpenAI(model_name="gpt-3.5-turbo",
                        temperature=0,
                        openai_api_key=openai_api_key),
                chain_type="stuff",
                retriever=docsearch.as_retriever(search_type="mmr") # Max Marginal Relevance Search
        )

        for q in questions:
            result = {'filename': filenames[doc_n], 'question': q}
            answer = chain({'question': f'{q}'}, return_only_outputs=True)
            result['answer'] = answer['answer']
            result['source'] = answer['sources']
            results.append(result)
    save_results(results)

def load_pdfs(directory_path: str) -> List[str]:
    pdfs = []
    filenames = []
    for file in os.listdir(directory_path):
        if file.endswith(".pdf"):
            path = os.path.join(directory_path, file)
            loader = PyPDFLoader(path)
            pdf = loader.load()
            pdfs.append(pdf)
            filenames.append(file)
    return pdfs, filenames

def load_questions(file_path):
    questions = []
    with open(file_path, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespaces and append to the list
            questions.append(line.strip())
    click.echo(questions)
    return questions

def save_token(token):
    # Set the environment variable with the provided token
    os.environ['OPENAI_API_KEY'] = token
    click.echo('Token saved as environment variable.')

def save_results(results):
    df = pd.DataFrame.from_dict(results)
    df.to_csv('results.csv', index=False)
    click.echo('Results saved!')

if __name__ == '__main__':
    extract_information()