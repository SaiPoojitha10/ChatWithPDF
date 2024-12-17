from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

pdf_file = "your_pdf_file.pdf"  
# Replace with the name of your PDF file
text = extract_text_from_pdf(pdf_file)
print("Extracted Text:", text)

from langchain.text_splitter import CharacterTextSplitter

# Split the text into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(text)

print("Number of Chunks:", len(chunks))
print("Sample Chunk:", chunks[0])

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Generate embeddings and store in ChromaDB
embeddings = OpenAIEmbeddings(openai_api_key="YOUR_OPENAI_API_KEY")
vector_store = Chroma.from_texts(chunks, embedding=embeddings)

print("Embeddings Created and Stored in Vector Database!")

query = input("Enter your query: ")  # User input
results = vector_store.similarity_search(query)

print("Query Results:")
for i, result in enumerate(results):
    print(f"Chunk {i+1}: {result.page_content}")