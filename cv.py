from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import Docx2txtLoader,PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import tempfile
import google.generativeai as genai
load_dotenv()
import textwrap
import streamlit as st
def process_docx(docx_file):
    text=""
    loader=Docx2txtLoader(docx_file)
    text=loader.load_and_split()
    return text
def process_pdf1(pdf_file):
    text=""
    loader=PyPDFLoader(pdf_file)
    pages=loader.load()
    for page in pages:
        text+=page.page_content
    text=text.repale('\t','')
    text_splitter=CharacterTextSplitter(separator="\n",chunk_size=100,chunk_overlap=20)
    docs=text_splitter.create_documents([text])
    return docs
def process_pdf(pdf_file):
    """
    Process a PDF file and return extracted text.
    """
    # Case 1: If pdf_file is a string -> assume it's a file path
    if isinstance(pdf_file, str):
        if not os.path.exists(pdf_file):
            raise ValueError(f"File path {pdf_file} does not exist.")
        loader = PyPDFLoader(pdf_file)
        return loader.load()

    # Case 2: If pdf_file is a Streamlit UploadedFile object
    if hasattr(pdf_file, "read"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path)
        return loader.load()

    raise TypeError("pdf_file must be a path string or file-like object.")
def main():
    st.title("CV Summary Generator")
    uploaded_file=st.file_uploader("Select CV",type=["docx","pdf"])
    text=""
    if uploaded_file is not None:
        fileExtension=uploaded_file.name.split('.')[-1]
        st.write("File Details:")
        st.write(f"File Name: {uploaded_file.name}")
        st.write(f"File Type: {fileExtension}")
        if fileExtension=="docx":
            text=process_docx(uploaded_file.name)
        elif fileExtension=="pdf":
            text=process_pdf(uploaded_file.name)
        else:
            st.error("Select supported format .pdf and docx")
            return
        llm=ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))
        prompt_template="""You have been the CV dails analysis the following
        {text}
        """
        prompt=PromptTemplate.from_template(prompt_template)
        refine_template="""
         "Your job is provide final outcome\n"
         "We have provided esisting details:{existing_answer} \n"
         "We want to refine the version of existing details based on initial details below\n"
         "_____________\n"
         "{text}\n"
         "_____________\n"
         "Give the new context,refine the original summary in the following manner \n"
         "Name: \n"
         "Email: \n"
         "Description: \n"
        """
        refine_prompt=PromptTemplate.from_template(refine_template)
        chain=load_summarize_chain(llm=llm,
                                   chain_type="refine",
                                   question_prompt=prompt,
                                   refine_prompt=refine_prompt,
                                   return_intermediate_steps=True,
                                   input_key="input_documents",
                                   output_key="output_texts"
                                   )
        result=chain.invoke({"input_documents":text},return_only_outputs=True)
        st.write("CV Summary")
        st.text_area("Text",result["text"],height=200)

if __name__ == "__main__":
    main()
