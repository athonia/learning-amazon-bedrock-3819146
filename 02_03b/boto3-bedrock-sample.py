#Import boto3 and json 
import streamlit
import boto3
import json

#Create the bedrock client
bedrock = boto3.client('bedrock-runtime',)

#Setting the prompt

# prompt_data = """Give me a detailed beat and bar structure for a melodic dubstep song. 
# Include Intro, Build-up, Drop, Breakdown, and Outro sections. 
# Format it with clear headings and describe the percussion, bass, melody, and effects in each section."""

prompt_data = """Give me a detailed beat and bar structure for a bass-house song. 
Include Intro, Build-up, Drop, Breakdown, and Outro sections. 
Format it with clear headings and describe the percussion, bass, melody, and effects in each section."""

#Model specification (Note that ChatGPT thinks these 3 below are best for discussing music structure)
#modelID = "amazon.titan-text-lite-v1"
modelID = "amazon.titan-text-express-v1"
#modelID = "anthropic.claude-instant-v1"
#modelID = "anthropic.claude-v2" #Best for music structure anthropic.claude-sonnet-4-20250514-v1:0
#modelID = "arn:aws:bedrock:us-east-1:609857695319:anthropic.claude-sonnet-4-20250514-v1:0"
#modelID = "anthropic.claude-3-5-haiku-20241022-v1:0"
accept = 'application/json'
contentType = 'application/json'
#Configuring parameters to invoke the model
body = json.dumps({
  "inputText": prompt_data, 
  "textGenerationConfig" : {
    "maxTokenCount": 1024
    },
    
  })
#Invoke the model
response = bedrock.invoke_model(
  body=body, modelId=modelID, accept=accept, contentType=contentType
  )
#Parsing and displaying the output
response_body = json.loads(response.get('body').read())
output = response_body['results'][0]['outputText']
print(output)