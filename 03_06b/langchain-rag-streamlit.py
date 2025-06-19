#Import libraries
import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain_community.chat_models import BedrockChat
from langchain.prompts.prompt import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

#Configure streamlit app
st.set_page_config(page_title="Social Media Training Bot", page_icon="📖")
st.title("📖 Social Media Training Bot")

#Define convenience functions
@st.cache_resource
def config_llm():
    client = boto3.client('bedrock-runtime')

    model_kwargs = { 
        "max_tokens": 512,
        "temperature":0.1,  
        "top_p":1
    }  

    model_id = "amazon.titan-text-lite-v1"
    llm = BedrockChat(
        model_id=model_id, 
        client=client,
        model_kwargs = model_kwargs
    )
    return llm

@st.cache_resource
def config_vector_db(filename):
    client = boto3.client('bedrock-runtime')
    embeddings_models_to_try = [
        "amazon.titan-text-lite-v1",
        "amazon.titan-text-express-v1",
        "anthropic.claude-3-haiku-20240307-v1:0"
    ]
    bedrock_embeddings = None # This was originally: BedrockEmbeddings(client=client)
for model_id in BedrockEmbeddings:
        try:
            bedrock_embeddings = BedrockEmbeddings(
                client=client,
                model_id=model_id
            )
            print(f"Successfully using embeddings model: {model_id}")
            break
        except Exception as e:
            print(f"Failed to use embeddings model {model_id}: {e}")
            continue
    
            if bedrock_embeddings is None:
                st.error("No embeddings model available. Check your AWS Bedrock permissions.")
            
        
            loader = PyPDFLoader(filename)
            pages = loader.load_and_split()
            vectorstore_faiss = FAISS.from_documents(pages, bedrock_embeddings)
            

#Configuring the llm and vector store
llm = config_llm()
vectorstore_faiss = config_vector_db("03_06b/social-media-training.pdf")

#Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")

if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")


#Creating the template   
my_template = """
Human: 
    You are a conversational assistant designed to help answer questions from an employee. 
    You should reply to the human's question using the information provided below. Include all relevant information but keep your answers short. Do not say things like "according to the training or handbook or based on or according to the information provided...".

<Information>
{info}
</Information>

{input}

Assistant:
"""

#Configure prompt template
prompt_template = PromptTemplate(
    input_variables=["info", "input"],
    template= my_template
)
#Create llm chain
question_chain = LLMChain(llm=llm, prompt=prompt_template, output_key= "answer")

#Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

#If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    # retrieve reevant documents with similarity search
    docs = vectorstore_faiss.similarity_search_with_score(prompt)
    info = ""
    for doc in docs:
        info += doc[0].page_content + "\n"
        #invoking LLM next
        output = question_chain.invoke({"info": info, "input": prompt})


#adding messages to history 
msgs.add_user_message(prompt)
msgs.add_ai_message(output["answer"])

# Display output
st.chat_message("ai").write(output["answer"])

    #Generate response
    #info = vectorstore_faiss.similarity_search(prompt, k=3)
    #response = question_chain.run(info=info, input=prompt)
    #st.chat_message("assistant").write(response)
    #msgs.add_ai_message(response)