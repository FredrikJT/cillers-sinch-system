from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
import os

app = Flask(__name__)

# Create a dictionary to store conversation chains per user
conversation_chains = {}

# Create an LLM instance with the updated OpenAI interface
llm = ChatOpenAI(model="gpt-4", api_key=os.environ.get("OPENAI_API_KEY"))

# Define the custom prompt template
template = """You are a helpful assistant that helps determine if a worker wants to cancel a shift and what shift they want to cancel.

If the worker wants to cancel a shift, confirm with them and ask for the specific hours of the shift if not provided.

Examples:

- Worker: "I can't come to work today."
  Assistant: "I'm sorry to hear that. Do you need to cancel your shift? If so, could you please provide the hours of the shift you need to cancel?"

- Worker: "I need to cancel my shift from 9am to 5pm tomorrow."
  Assistant: "Understood, you need to cancel your shift from 9am to 5pm tomorrow. I'll process that for you."

Use the conversation history to maintain context. Be polite and helpful in your responses.

{history}
Worker: {input}
Assistant:"""

prompt = PromptTemplate(input_variables=["history", "input"], template=template)

# Chain prompt with LLM and StrOutputParser
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_id = data.get('user_id')
    message = data.get('message')

    if not user_id or not message:
        return jsonify({'error': 'user_id and message are required'}), 400

    # Create or retrieve memory and chain for the user
    if user_id not in conversation_chains:
        memory = ConversationBufferMemory(memory_key="history")
        conversation_chains[user_id] = chain
    else:
        # If user exists, retrieve their chain and memory
        memory = ConversationBufferMemory(memory_key="history")

    # Get response from the conversation chain
    memory_variables = memory.load_memory_variables({"input": message})
    response = conversation_chains[user_id].invoke({"input": message, "history": memory_variables["history"]})

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
