import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import warnings, os
from dotenv import load_dotenv
warnings.filterwarnings("ignore")
__import__('sqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('sqlite3')


# import sqlite3
# conn = sqlite3.connect('example.db')
# c = conn.cursor()


# Load environment variables from .env file
load_dotenv()

data_directory = os.path.join(os.path.dirname(__file__), "data")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# st.secrets["huggingface_api_token"] # Don't forget to add your hugging face token

# Load the vector store from disk
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(embedding_function=embedding_model, persist_directory=data_directory)

# Initialize the Hugging Face Hub LLM
hf_hub_llm = HuggingFaceHub(
     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    # repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    model_kwargs={"temperature": 1, "max_new_tokens":1024},
)

prompt_template = """
As a highly knowledgeable medical assistant, your role is to accurately interpret medical queries and 
provide responses using our specialized medical database. Follow these directives to ensure optimal user interactions:
1. Precision in Answers: Respond solely with information directly relevant to the user's query from our medical database. 
   Refrain from making assumptions or adding extraneous details.
2. Topic Relevance: Limit your expertise to specific medical-related areas:
   - Medical Prescriptions
   - Medical Advice
   - Symptom Analysis
   - Treatment Recommendations
   - Preventive Healthcare
3. Handling Off-topic Queries: For questions unrelated to medical topics (e.g., general knowledge questions like "Why is the sky blue?"), 
   politely inform the user that the query is outside the chatbot’s scope and suggest redirecting to medical-related inquiries.
4. Promoting Health Awareness: Craft responses that emphasize good health practices, aligning with the latest medical guidelines and 
   personalized health recommendations.
5. Contextual Accuracy: Ensure responses are directly related to the medical query, utilizing only pertinent 
   information from our database.
6. Relevance Check: If a query does not align with our medical database, guide the user to refine their 
   question or politely decline to provide an answer.
7. Avoiding Duplication: Ensure no response is repeated within the same interaction, maintaining uniqueness and 
   relevance to each user query.
8. Streamlined Communication: Eliminate any unnecessary comments or closing remarks from responses. Focus on
   delivering clear, concise, and direct answers.
9. Avoid Non-essential Sign-offs: Do not include any sign-offs like "Best regards" or "MedicalBot" in responses.
10. One-time Use Phrases: Avoid using the same phrases multiple times within the same response. Each 
    sentence should be unique and contribute to the overall message without redundancy.

Medical Query:
{context}

Question: {question}

Answer:

"""

# prompt_template="""
# Use the following pieces of information to answer the user's question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.

# Context: {context}
# Question: {question}

# Only return the helpful answer below and nothing else.
# Helpful answer:
# """

custom_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

rag_chain = RetrievalQA.from_chain_type(
    llm=hf_hub_llm, 
    chain_type="stuff", 
    retriever=vector_store.as_retriever(top_k=3),  # retriever is set to fetch top 3 results
    chain_type_kwargs={"prompt": custom_prompt})

def get_response(question):
    result = rag_chain({"query": question})
    response_text = result["result"]
    answer_start = response_text.find("Answer:") + len("Answer:")
    answer = response_text[answer_start:].strip()
    return answer

# Streamlit app
# Remove whitespace from the top of the page and sidebar
st.markdown(
        """
            <style>
                .appview-container .main .block-container {{
                    padding-top: {padding_top}rem;
                    padding-bottom: {padding_bottom}rem;
                    }}

            </style>""".format(
            padding_top=1, padding_bottom=1
        ),
        unsafe_allow_html=True,
    )

# st.header("### Discover the AI Medical Recommendations 💉🩺 ", divider='grey')
st.markdown("""
    <h3 style='text-align: left; color: white ; padding-top: 35px; border-bottom: 3px solid red;'>
        Discover the AI Medical Recommendations 💉🩺\n
    </h3>""", unsafe_allow_html=True)
st.write('Made with ❤️ by [Pratik](https://github.com/Pratikgirade)')


side_bar_message = """
Hi! 👋 I'm here to help you with your medical queries and give you Prescription. What would you like to know or explore?
\nHere are some areas you might be interested in:
1. **General Health Tips** 🏥
2. **Symptom Analysis** 🤒
3. **Medication Information** 💊
4. **Preventive Care** 🛡️

Feel free to ask me anything about your health!
"""

with st.sidebar:
    st.title('🤖MedBot: Your AI Health Companion')
    st.markdown(side_bar_message)

initial_message = """
    Hi there! I'm your MedBot 🤖 
    Here are some questions you might ask me:\n
     🩺Provide me tablets for "<Symptoms you are facing>" and explain me its dose process?\n
     🩺What are the best practices to boost my immune system?\n
     🩺Can you help me understand my "<symptom>"?\n
     🩺What are the common side effects of this Tablet?\n
     🩺Provide me tablets for headache?\n
     🩺What's the best approach for managing chronic pain?
"""

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]
st.button('Clear Chat', on_click=clear_chat_history)

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Hold on, I'm fetching the latest medical advice for you..."):
            response = get_response(prompt)
            placeholder = st.empty()
            full_response = response  # Directly use the response
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
