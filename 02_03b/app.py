
import streamlit as st
import boto3
import json

# Streamlit UI
st.title("An AI Song Structure Generator")
prompt_data = st.text_area("Hint: Tell the model who they are and where they have expertise before asking questions. Example:", height=200, value="""You are a musicologist and veteran producer of dance beats. 
Give me a detailed beat and bar structure for an EDM song in the genre of: 

ENTER THE MUSIC GENRE HERE e.g., melodic dubstep, bass-house, cottage-core, etc.

Include Intro, Build-up, Drop, Breakdown, and Outro sections.
Format it with clear headings and describe the percussion, bass, melody, and effects in each section.""")

if st.button("Generate"):
  with st.spinner("Generating response..."):
    try:
        # Create the Bedrock client
        bedrock = boto3.client('bedrock-runtime')

        # Model configuration
        modelID = "amazon.titan-text-express-v1"
        accept = 'application/json'
        contentType = 'application/json'

        # Prepare the request body
        body = json.dumps({
            "inputText": prompt_data,
            "textGenerationConfig": {
                "maxTokenCount": 1024
            },
        })

        # Invoke the model
        response = bedrock.invoke_model(
            body=body,
            modelId=modelID,
            accept=accept,
            contentType=contentType
        )

        # Parse and display the output
        response_body = json.loads(response.get('body').read())
        output = response_body['results'][0]['outputText']
        st.subheader("Generated Output")
        st.write(output)

    # Removed redundant except block

    except Exception as e:
          st.error(f"An error occurred: {e}")