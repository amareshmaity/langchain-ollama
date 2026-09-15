from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pathlib import Path


## load + split + embed + store
BASE_DIR = Path(__file__).resolve().parent
path_of_pdf_file = str(BASE_DIR.parent  / "pdf" / "LangChain.pdf")
loader = PyPDFLoader(path_of_pdf_file)
document = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents=document)
store = Chroma.from_documents(documents=chunks, embedding=OllamaEmbeddings(model="mxbai-embed-large:latest"))

retriver = store.as_retriever(search_kwargs={"k":3})

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question only using the context below: \n\n {context}"),
    ("human", "{question}")
])

model = ChatOllama(model="deepseek-r1:1.5b")
parser = StrOutputParser()

chain = (
    {"context":retriver, "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)


user_input = input(f"\n\n{'-'*30}\nAsk anything from the pdf file\n{"-"*30}\nQuestion: ")
while user_input:
    if user_input == "bye":
        break
    print("\nAnswer:")
    print(chain.invoke(user_input))
    user_input = input(f"\n\n{'-'*40}\nAsk next question or type 'bye' to exit\n{'-'*40}\n")
