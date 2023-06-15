from unstructured.partition.auto import partition
link="/home/jam/Downloads/3HAC050917 TRM RAPID RW 6-en.pdf";
elements = partition(filename=link,content_type="application/pdf")

from unstructured.documents.elements import NarrativeText
from unstructured.partition.text_type import sentence_count
from langchain.text_splitter import NLTKTextSplitter

input=""
for element in elements:
    if isinstance(element, NarrativeText):
        input=input+str(element);
        
       
nltksplit = NLTKTextSplitter(chunk_size=250)
nsplit = nltksplit.split_text(input) 

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import os
os.environ["OPENAI_API_KEY"] = 'YOUR_OPENAI_APIKEY'

embed = OpenAIEmbeddings()

db = Chroma.from_texts(nsplit, embed)

query = "how to define a boolean"
outputtext = db.similarity_search(query)
print(outputtext[0].page_content)

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)

from langchain.llms import OpenAI
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), db.as_retriever(), memory=memory)

query = "What is this manual about?"
result = qa({"question": query})
result["answer"]

query = "tell me how do I move my robot linearly and circularly"
result = qa({"question": query})
result["answer"]
