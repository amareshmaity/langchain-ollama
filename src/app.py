from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pathlib import Path

## Render raw output
import re
from pylatexenc.latex2text import LatexNodes2Text
from rich.console import Console
from rich.markdown import Markdown

def render_deepseek_output(raw_ai_text: str):
    """
    Parses DeepSeek raw text, converts internal LaTeX tokens into plain Unicode math,
    and prints a beautifully rendered Markdown interface to the terminal.
    """
    # Create the converter tool for LaTeX formulas
    # math_mode='text' formats symbols cleanly; fill_text=False maintains original layouts
    tex_converter = LatexNodes2Text(math_mode='text', fill_text=False)
    
    # 1. Standardize DeepSeek's display math delimiters [ \int ... ] into standard $$ \int ... $$
    processed_text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', raw_ai_text, flags=re.DOTALL)
    
    # 2. Standardize inline math delimiters ( x^3 ) or \( x^3 \) into standard $x^3$
    processed_text = re.sub(r'\\\((.*?)\\\)', r'$\1$', processed_text)
    
    # 3. Find and extract math blocks to convert them via pylatexenc
    def latex_replacer(match):
        latex_code = match.group(1)
        try:
            # Safely convert equations into Unicode strings (e.g., \frac{1}{2} -> 1/2)
            converted_math = tex_converter.latex_to_text(latex_code)
            return f"**{converted_math.strip()}**" # Bold the formula for visibility
        except Exception:
            return match.group(0) # Fallback to original text if parsing fails
            
    # Apply conversion to both display blocks ($$) and inline math ($)
    processed_text = re.sub(r'\$\$(.*?)\$\$', latex_replacer, processed_text, flags=re.DOTALL)
    processed_text = re.sub(r'\$(.*?)\$', latex_replacer, processed_text)

    # 4. Hand the stylized string to Rich to display cleanly in the terminal window
    console = Console()
    rendered_markdown = Markdown(processed_text)
    console.print(rendered_markdown)





## load + split + embed + store
BASE_DIR = Path(__file__).resolve().parent
path_of_pdf_file = str(BASE_DIR.parent  / "pdf" / "lecture-notes-on-integration.pdf")
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
    # print(chain.invoke(user_input))
    response = chain.invoke(user_input)
    render_deepseek_output(response)
    user_input = input(f"\n\n{'-'*40}\nAsk next question or type 'bye' to exit\n{'-'*40}\n")
