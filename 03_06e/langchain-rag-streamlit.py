#Import libraries
import boto3
from langchain_aws import BedrockEmbeddings
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
        "maxTokenCount": 512,
        "temperature":0.1,  
        "topP":1
    }  
    model_id = "amazon.titan-text-express-v1"
 #   model_id = "anthropic.claude-instant-v1"
    llm = Bedrock(model_id=model_id, client=client)
    llm.model_kwargs = model_kwargs
    return llm

@st.cache_resource
def config_vector_db(filename):
    client = boto3.client('bedrock-runtime')
    
    # Try both embeddings models
    embeddings_models = [
        "amazon.titan-embed-text-v1",
        "amazon.titan-embed-text-v2:0"
    ]
    
    for model_id in embeddings_models:
        try:
            bedrock_embeddings = BedrockEmbeddings(
                client=client,
                model_id=model_id
            )
            st.success(f"Using embeddings model: {model_id}")
            break
        except Exception as e:
            st.warning(f"Failed to access {model_id}: {str(e)}")
            continue
    else:
        st.error("No embeddings model available")
        return None
    
    loader = PyPDFLoader(filename)
    pages = loader.load_and_split()
    vectorstore_faiss = FAISS.from_documents(pages, bedrock_embeddings)
    return vectorstore_faiss

#Configuring the llm and vector store
llm = config_llm()
vectorstore_faiss = config_vector_db("03_06e/social-media-training.pdf")

# Add this check
if vectorstore_faiss is None:
    st.error("Cannot proceed without embeddings access. Please wait for model approval.")
    st.stop()

#Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0 :
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
    input_variables= ['info', 'input'],
    template= my_template
)

#Create llm chain
question_chain = LLMChain(
    llm = llm,
    prompt = prompt_template,
    output_key= "answer"
)

#Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

#If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)

    #retrieve relevant documents using a similarity search
    docs = vectorstore_faiss.similarity_search_with_score(prompt)
    info = ""
    for doc in docs:
        info += doc[0].page_content + '\n'

    #invoke llm
    output = question_chain.invoke({"input" : prompt, "info" : info})

    #adding messages to history
    msgs.add_user_message(prompt)
    msgs.add_ai_message(output['answer'])

    #display the output
    st.chat_message("ai").write(output['answer'])
