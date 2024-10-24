from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
import os

app = Flask(__name__)

# Create a dictionary to store conversation chains per user
conversation_chains = {}

# Create an LLM instance with the updated OpenAI interface
llm = ChatOpenAI(model="gpt-4", api_key=os.environ.get("OPENAI_API_KEY"))

DBSTR = "USERID | SHIFT  | START_DATE | START_TIME | END_TIME\n001 | Shift1 | 2024-10-01 | 09:00 | 17:00\n002 | Shift2 | 2024-10-01 | 10:00 | 18:00"

# Define the custom prompt template
template = """You are a helpful assistant that helps determine if a worker wants to cancel a shift and what shift they want to cancel.

If the worker wants to cancel a shift, confirm with them and ask for the specific hours of the shift if not provided.

You will be given data from a database. Use that to answer the questions:  

-- DATABASE -- 
{database}

Use the conversation history to maintain context. Be polite and helpful in your responses.

-- CONVERSATION HISTORY -- 
{history}
Worker: {input}
Assistant:"""

prompt = PromptTemplate(input_variables=["history", "input", "database"], template=template)

# Chain prompt with LLM and StrOutputParser for conversation handling
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Define a separate prompt for updating the database
db_prompt_template = """Generate a SQL command to update the worker's shift in the database based on the following conversation:

Conversation: {conversation}

--DATABASE -- 
{database}

Provide the SQL statement.
"""

db_prompt = PromptTemplate(input_variables=["conversation"], template=db_prompt_template)

# Custom SQL output parser that checks for valid SQL statements
class SQLResultOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        # Basic validation to check if the output contains a valid SQL statement
        if "SELECT" in text or "INSERT" in text or "UPDATE" in text or "DELETE" in text:
            return text.strip()
        else:
            return None  # Return None if no valid SQL command is found

    def get_format_instructions(self) -> str:
        return "Please provide a valid SQL statement."

# Chain to handle database updates
update_db_chain = db_prompt | llm | SQLResultOutputParser()

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

    # Get the conversation response
    memory_variables = memory.load_memory_variables({"input": message})
    response = conversation_chains[user_id].invoke({
        "input": message,
        "history": memory_variables["history"],
        "database": DBSTR
    })

    # Feed the response into the update_db_chain to generate the SQL command
    db_response = update_db_chain.invoke({"conversation": response, "database":DBSTR})

    return jsonify({'response': response, 'db_response': db_response})

if __name__ == '__main__':
    app.run(debug=True)
