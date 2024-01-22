__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import click
import sys
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI
import pandas as pd

@click.command()
@click.option('--input-directory', type=click.Path(exists=True), required=True, help='The input directory containing PDF files.')
@click.option('--questions-file', type=click.Path(exists=True), required=True, help='The questions file. Each question should be in a new line.')
@click.option('--openai-api-key', type=click.STRING, envvar='OPENAI_API_KEY', required=True, help='The OpenAI API key.')

def extract_pdf_information(input_directory, questions_file, openai_api_key):
    """
    Extract information from PDF files using provided questions and API key.
    """
    click.echo(f"Loading PDF files from '{input_directory}'...")
    pdf_documents, filenames = load_pdf_files(input_directory)
    click.echo(f"Loaded {len(pdf_documents)} PDF files.")

    click.echo(f"Loading questions from '{questions_file}'...")
    questions_list = load_questions_from_file(questions_file)
    click.echo(f"Loaded {len(questions_list)} questions.")

    MAX_TOKENS = 512  # Maximum number of tokens allowed by the model
    chunk_size = MAX_TOKENS - 100  # Subtracting a buffer of tokens for safety
    chunk_overlap = chunk_size // 20  # Using 10% overlap
    document_chunks = []

    for pdf in pdf_documents:
        """
        Split the PDF documents into smaller chunks to handle the maximum token limit.
        """
        click.echo(f"Splitting PDF '{filenames[pdf_documents.index(pdf)]}'...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                        chunk_overlap=chunk_overlap)
        document_chunks.extend(text_splitter.split_documents(pdf))

    results = []
    for doc_index, document in enumerate(document_chunks):
        """
        Create embeddings for the documents and build a vector store for efficient searching.
        """
        click.echo(f"Creating embeddings for document chunk {doc_index + 1}...")
        embeddings = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
        document_search = Chroma.from_documents([document], embeddings)

        document.metadata['source'] = f"chunk-{doc_index}"

        chain = RetrievalQAWithSourcesChain.from_chain_type(
                ChatOpenAI(model_name="gpt-3.5-turbo",
                           temperature=0,
                           openai_api_key=openai_api_key),
                chain_type="stuff",
                retriever=document_search.as_retriever(search_kwargs={"k": 1})
        )

        for question in questions_list:
            """
            Process the questions and generate answers using the chain.
            """
            click.echo(f"Processing question '{question}'...")
            result = {'filename': filenames[doc_index], 'question': question}
            answer = chain.invoke({'question': f'{question}'}, return_only_outputs=True)
            result['answer'] = answer['answer']
            result['source'] = answer['sources']
            results.append(result)

    save_results_to_csv(results)

def load_pdf_files(input_directory: str) -> List[str]:
    """
    Load PDF files from the input directory and return the list of PDF documents and filenames.
    """
    pdf_files = []
    filenames = []
    for file in os.listdir(input_directory):
        if file.endswith(".pdf"):
            path = os.path.join(input_directory, file)
            loader = PyPDFLoader(path)
            pdf_files.append(loader.load())
            filenames.append(file)
    return pdf_files, filenames

def load_questions_from_file(questions_file):
    """
    Load questions from the questions file and return the list of questions.
    """
    questions = []
    with open(questions_file, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespaces and append to the list
            questions.append(line.strip())
    click.echo(questions)
    return questions

def save_results_to_csv(results):
    """
    Save the results to a CSV file.
    """
    df = pd.DataFrame.from_dict(results)
    df.to_csv('results.csv', index=False)
    click.echo('Results saved!')

if __name__ == '__main__':
    extract_pdf_information()