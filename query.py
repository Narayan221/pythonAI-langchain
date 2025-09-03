from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# Load vector DB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="db", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 1})

# Load local LLM
pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=200)
llm = HuggingFacePipeline(pipeline=pipe)

# Prompt: only the answer
prompt_template = """
You are a precise resume parser.

Context:
{context}

Question:
{question}

Instructions:
- Extract only the relevant information.
- If the question is vague (like "info", "details"), return a summary of all key fields: Name, Email, Phone, Experience, Education, Skills.
- If the information is missing, reply with "Not available".
- Return only the answer in a single line.
"""


prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# Query loop
while True:
    query = input("\nAsk something about resumes (or 'exit'): ")
    if query.lower() == "exit":
        break
    answer = qa.invoke({"query": query})
    print("Answer:", answer["result"].strip())
