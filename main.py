from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, WebBaseLoader, YoutubeLoader, DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI
import numpy as np
import tiktoken
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()



pinecone_api_key = os.environ.get("PINECONE_API_KEY")

groq_api_key = os.environ.get("GROQ_API_KEY")


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")





text = "Hello my name is Faizan"

query_result = embeddings.embed_query(text)


print(len(query_result))



# Free Llama 3.1 API via Groq

groq_client = Groq(api_key=groq_api_key)












def get_huggingface_embeddings(text, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)


def cosine_similarity_between_sentences(sentence1, sentence2):
    # Get embeddings for both sentences
    embedding1 = np.array(get_huggingface_embeddings(sentence1))
    embedding2 = np.array(get_huggingface_embeddings(sentence2))

    # Reshape embeddings for cosine_similarity function
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)

    print("Embedding for Sentence 1:", embedding1)
    print("\nEmbedding for Sentence 2:", embedding2)

    # Calculate cosine similarity
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity[0][0]


# Example usage
sentence1 = "I like walking to the park"
sentence2 = "I like running to the office"


similarity = cosine_similarity_between_sentences(sentence1, sentence2)
print(f"\n\nCosine similarity between '{sentence1}' and '{sentence2}': {similarity:.4f}")





def process_directory(directory_path):
    data = []
    for root, _, files in os.walk(directory_path):
        for file in files:

            file_path = os.path.join(root, file)
            #print(f"Processing file: {file_path}")
            loader = PyPDFLoader(file_path)
            data.append({"File": file_path, "Data": loader.load()})

    return data

directory_path = "./CompanyDocuments"
documents = process_directory(directory_path)





# Make sure to create a Pinecone index with 384 dimensions

index_name = "rag-experiment"

namespace = "company-documents"

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)







# Prepare the text for embedding
document_data = []
for document in documents:

    document_source = document['Data'][0].metadata['source']
    document_content = document['Data'][0].page_content

    file_name = document_source.split("/")[-1]
    folder_names = document_source.split("/")[2:-1]

    doc = Document(
        page_content = f"\n{document_source}\n\n\n\n{document_content}\n",
        metadata = {
            "file_name": file_name,
            "parent_folder": folder_names[-1],
            "folder_names": folder_names
        }
    )
    document_data.append(doc)
     



# Insert documents into Pinecone
vectorstore_from_documents = PineconeVectorStore.from_documents(
    document_data,
    embeddings,
    index_name=index_name,
    namespace=namespace
)




# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Connect to your Pinecone index
pinecone_index = pc.Index(index_name)




query = "What are some items that Pirkko Koskitalo is likely to buy next? What incentives can I put in place to ensure he orders more?"
     


raw_query_embedding = get_huggingface_embeddings(query)





top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k=10, include_metadata=True, namespace=namespace)
     
contexts = [item['metadata']['text'] for item in top_matches['matches']]


augmented_query = "\n" + "\n\n-------\n\n".join(contexts[ : 10]) + "\n-------\n\n\n\n\nMY QUESTION:\n" + query
     


system_prompt = f"""You are an expert at understanding and analyzing company data - particularly shipping orders, purchase orders, invoices, and inventory reports.

Answer any questions I have, based on the data provided. Always consider all of the context provided when forming a response.
"""

llm_response = groq_client.chat.completions.create(
    model="llama-3.1-70b-versatile",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": augmented_query}
    ]
)

response = llm_response.choices[0].message.content





def perform_rag(query):
    raw_query_embedding = get_huggingface_embeddings(query)

    query_embedding = np.array(raw_query_embedding)

    top_matches = pinecone_index.query(vector=query_embedding.tolist(), top_k=10, include_metadata=True, namespace=namespace)

    # Get the list of retrieved texts
    contexts = [item['metadata']['text'] for item in top_matches['matches']]

    augmented_query = "\n" + "\n\n-------\n\n".join(contexts[ : 10]) + "\n-------\n\n\n\n\nMY QUESTION:\n" + query

    # Modify the prompt below as need to improve the response quality
    system_prompt = f"""You are an expert at understanding and analyzing company data - particularly shipping orders, purchase orders, invoices, and inventory reports.

    Answer any questions I have, based on the data provided. Always consider all parts of the context provided when forming a response.
    """

    res = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile", # llama-3.1-70b-versatile
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    return res.choices[0].message.content


response = perform_rag("What are some trends with Ricardo Adocicados purchase orders?")

print(response)