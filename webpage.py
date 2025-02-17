import streamlit as st
import time
import uuid
import os
import logging
from utils import retriever, get_session_history, rag_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from PIL import Image
import re
from config import Config




# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG,  # You can change this to DEBUG or ERROR as needed
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # Log to a file
        logging.StreamHandler()  # Also log to the console
    ]
)

SESSION_TIMEOUT = 1800  # 30 minutes


def reset_session():
    """Reset the session state, including session ID, messages, and interaction time."""
    st.session_state.clear()
    st.session_state.session_id = str(uuid.uuid4())  # Create a new session ID
    st.session_state.messages = []  # Reset chat history
    st.session_state.last_interaction = time.time()  # Reset last interaction time


# Initialize session if not set or reset if session timed out
if 'session_id' not in st.session_state:
    reset_session()

else:
    # Initialize last interaction time if it's not already set
    current_time = time.time()
    last_interaction = st.session_state.get('last_interaction', current_time)
    
    if current_time - last_interaction > SESSION_TIMEOUT:
        # Reset session if the user has been inactive for longer than the timeout
        reset_session()
    else:
        st.session_state.last_interaction = current_time



st.set_page_config(page_title="Mercy LucIA", layout="wide", page_icon='❣️')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add custom CSS with Clipelvic brand colors
st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-color: #FFFFFF;
        }

        
        /* Headers and important text */
        h1, h2, h3, h4, h5 {
            color: #4B0082 !important;
        }
        
        /* Buttons and interactive elements */
        .stButton>button {
            background-color: #9932CC;
            color: white;
        }
        
        /* Links */
        a {
            color: #9932CC !important;
        }
    </style>
""", unsafe_allow_html=True)




with st.container():
  
      # Center column for the image
      # Use full width of the column

    # Create a three-column layout for the text box
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Display the centered text box with reduced spacing
        st.image("LogoMercy1.png", width = 100, use_column_width= True)
        st.markdown(
            """
            <div style='text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: -10px; padding: 0px; width: 100%;'>
                <h2>¡Hola! Soy Mercy LucIA</h2>
                <h5>Tu Especialista Virtual Artificial</h5>  
                <h5>¡Estoy aquí para resolver tus dudas!</h5>
            </div>
            """,
            unsafe_allow_html=True
        )


# Sidebar functionalities
def sidebar():
    
    
    
    # Further Assistance Information
    st.sidebar.markdown("###  ¿Necesitas más información?")
       
    st.sidebar.markdown(
    """
    
    
    ▶️ <a href="https://wa.me/573008208719" target="_blank">Contáctanos</a>  <br>
    
 
    📍 Cra 17a #15 - 92, CliPelvic, Barrio San Vicente, Valledupar
    """,
    unsafe_allow_html=True
)
    st.sidebar.markdown(
    """
    ---
    <div style='font-size: 16px;'>  <!-- Adjust the font size here -->
        <strong> Disclaimer:</strong> <br>
        Esta información es proporcionada por un asistente virtual para fines educativos. 
        No sustituye el consejo médico profesional. 
        Para una evaluación médica adecuada, consulta a tu médico.
    </div>
    """,
    unsafe_allow_html=True
)
    

    st.sidebar.markdown(
    """
    <div style='text-align: center; margin-top: 18px;'>
        <p style='color: gray;'> 2025, Made with 🖤 by Dialogik.co </p>
    </div>
    """,
    unsafe_allow_html=True
)
    # Disclaimer in Sidebar
    


# Main chat application
def chat():
    inline_style = "font-size: 18px !important; line-height: 1.6 !important; font-family: Arial, sans-serif !important; font-weight: normal !important;"

    st.markdown(
    """<style> 
    .response-text { 
        font-size: 18px !important; 
        line-height: 1.6; 
        font-family: Arial, sans-serif !important;
        font-weight: normal !important;
    } 
    </style>""",
    unsafe_allow_html=True
)
  
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            content = message["content"]
            st.markdown(f"<div style='{inline_style}'>{content}</div>", unsafe_allow_html=True)


    # Chat input field for user input
    user_message = st.chat_input("Escribe tu mensaje aquí...")

    if user_message:
        # Store and display the user message
        st.session_state.messages.append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            st.markdown(f"<div style='{inline_style}'>{user_message}</div>", unsafe_allow_html=True)

      
        common_responses = {
    "hola": "Hola Corazón 💓. Soy Mercy LucIA, Especialista Virtual con Inteligencia Artificial de la Dra. Mercy. Estoy aquí para ayudarte a resolver tus dudas sobre ginecología, sexualidad, piso pélvico, ginecología estética, y más ❤️ ",
    "gracias": "¡Con gusto! Si tienes otra consulta, estoy aquí para ayudarte 😊",
    "¿quién te creó?": "Fui creado por Dialogik, un equipo de expertos en tecnología y automatización para ayudarte en todo lo que necesites 📚🔧. Sin embargo, si se trata de una consulta médica, es mejor que la consultes directamente a ella. Soy un sistema de Ineligencia artificial y puedo cometer errores.",
    "¿cuál es tu nombre?": "Mi nombre es Mercy LucIA 💜, tu asistente virtual siempre disponible para ayudarte.",
    "¿qué puedes hacer?": "Estoy entrenado para responder preguntas que puedas tener en relación con la preparación para cirugías para que estés lo mejor informado posible!",
    "¿eres un robot?": "Sí, soy un robot asistente virtual diseñado para ayudarte con información y consultas 👾",
    "¿trabajas las 24 horas?": "¡Así es! Estoy disponible las 24 horas del día, los 7 días de la semana, siempre listo para ayudarte 💪",}




        response_text = common_responses.get(user_message.lower(), None)
        if response_text:
            with st.chat_message("assistant"):
                st.markdown(f"<div style='{inline_style}'>{response_text}</div>", unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                response_placeholder = st.empty()  # Placeholder for the response
                response_placeholder.markdown(f"<div style='{inline_style}'> ... </div>", unsafe_allow_html=True)

            try:
                print("I should see this in the terminal")
                logging.debug(f"Hello, I am trapped inside your program")
                conversational_rag_chain = RunnableWithMessageHistory(
                    rag_chain,
                    get_session_history,
                    input_messages_keys='input',
                    history_messages_key='chat_history',
                    output_messages_key='answer'
                )
                docs = retriever.invoke(user_message)

                if not docs:
                    raise ValueError("No documents found by retriever.")

                context_text = "\n\n".join([(doc.page_content) for doc in docs])

                response = conversational_rag_chain.invoke(
                    {"context": context_text, "input": user_message},
                    config={"configurable": {"session_id": st.session_state.session_id}}
                )

                if 'answer' not in response:
                    raise KeyError("Response missing 'answer' key.")
                
                response_text = response.get('answer')
            except KeyError as ke:
                logging.error(f"KeyError encountered: {ke}")
                response_text = "En el momento estamos expermientando algunos problemas 💔 . Volveremos a estar disponibles en breve."
            except ValueError as ve:
                logging.error(f"ValueError encountered: {ve}")
                response_text = "En el momento estamos expermientando algunos problemas 💔 . Volveremos a estar disponibles en breve."

            except Exception as e:
                logging.error(f"An unexpected error occurred: {str(e)}")
                response_text = "En el momento estamos expermientando algunos problemas 💔 . Volveremos a estar disponibles en breve."
                # Display the assistant's response
                
            
            response_placeholder.markdown(f"<div style='{inline_style}'>{response_text}</div>", unsafe_allow_html=True)

           

      

        
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        
      

    # Add disclaimer at the bottom of the chat if no messages yet
    
    if len(st.session_state.messages)== 0:
      st.markdown(
    """
    <div style='font-size: 18px; line-height: 1.6;'>  
        <strong>Disclaimer:</strong>  
        Las respuestas proporcionadas por este asistente virtual son de carácter informativo y no reemplazan una consulta médica. 
        Para un diagnóstico o tratamiento adecuado, por favor consulta a la Dra. Mercy Dávila.
    </div>
    """,
    unsafe_allow_html=True
)
    
# Main App
def main():
    sidebar()  # Load the sidebar functionalities
    chat()  # Load the main chat interface

if __name__ == "__main__":
    main()


