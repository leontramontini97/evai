import streamlit as st
import pandas as pd
from datetime import datetime, timezone, timedelta
import time
import boto3
import os

import plotly.express as px

# Load environment variables


# Initialize DynamoDB
dynamodb = boto3.resource('dynamodb', region_name='us-east-2')
table = dynamodb.Table('chaatbot_mercy')

# Load data from DynamoDB
def load_conversations():
    try:
        response = table.scan()
        items = response['Items']
        
        # Convert to DataFrame
        df = pd.DataFrame(items)
        
        # Convert timestamp strings to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading conversations: {e}")
        return pd.DataFrame()

#######################
# Login functionality
USERNAME = "dramercy"
PASSWORD = "Clipelvic.2025"

def login():
    """Handle the login form."""
    st.title("Login")

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if st.session_state["logged_in"]:
        st.write("Ya has iniciado sesión. Ve al panel de mensajes.")
    else:
        username = st.text_input("Nombre de usuario")
        password = st.text_input("Contraseña", type="password")
        if st.button("Login"):
            if username == USERNAME and password == PASSWORD:
                st.session_state["logged_in"] = True
                st.session_state.query_params = st.query_params
                st.session_state.query_params.update({'logged_in': 'true'})
                st.write("Inicio de sesión exitoso!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Credenciales incorrectas")

def logout():
    """Handle the logout action."""
    st.session_state["logged_in"] = False
    st.rerun()

def display_dashboard():
    """Display the main message dashboard if logged in."""
    st.title("Mensajes Recibidos")
    

    
    # Load messages from DynamoDB
    df_messages = load_conversations()

    if not df_messages.empty:
        # Sort by timestamp in descending order (newest first)
        df_messages = df_messages.sort_values(by='timestamp', ascending=False)
        
        st.markdown("### Último Mensaje")
        latest_message = df_messages.iloc[0]  # Get the first row (most recent) after sorting

        st.markdown(f"""
         <div>
            <p style='font-size:14px;'><strong>Mensaje:</strong> {latest_message['message_body']}</p>
            <p style='font-size:14px;'><strong>Fecha y Hora:</strong> {latest_message['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)

        # Display messages table
        st.markdown("### Tabla con mensajes ")

        # Ensure correct columns are present
        required_columns = ['id', 'message_body', 'response', 'category', 'timestamp', 'session_id']
        df_messages = df_messages.reindex(columns=required_columns)

        # Sort DataFrame by timestamp in descending order (newest first)
        df_messages = df_messages.sort_values(by='timestamp', ascending=False)

        # Display the DataFrame
        st.dataframe(df_messages, hide_index=True, column_config={
            "id": st.column_config.TextColumn("ID"),
            "message_body": st.column_config.TextColumn("Mensaje"),
            "response": st.column_config.TextColumn("Respuesta"),
            "timestamp": st.column_config.TextColumn("Fecha y Hora"),
            "session_id": st.column_config.TextColumn("Session ID"),
        })

        # Category distribution pie chart
        df_messages['category'] = df_messages['category'].fillna('Desconocido')
        category_counts = df_messages['category'].value_counts()
        fig = px.pie(values=category_counts.values, names=category_counts.index, 
                    title="Distribución de Categorías de Preguntas")
        st.plotly_chart(fig)

        # Detailed message view
        st.markdown("### Todos los mensajes ")
        df_messages_sorted = df_messages.sort_values(by='timestamp', ascending=False)
        
        for idx, row in df_messages_sorted.iterrows():
            with st.expander(f"Mensaje {row['id']}"):
                st.write(f"**Pregunta:** {row['message_body']}")
                st.write(f"**Respuesta:** {row['response']}")
                st.write(f"**Categoría:** {row['category']}")
                st.write(f"**Fecha y Hora:** {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

    # Sidebar
    with st.sidebar:
        st.sidebar.image('LogoMercy2.png', use_container_width=True)
        st.title("Panel de Mensajes")
        if st.button("Cerrar sesión"):
            logout()
        st.sidebar.markdown(
            """
            <div style='text-align: center; margin-top: 18px;'>
                <p style='color: gray;'>Made with 🖤 by Dialogik.co, 2025. </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Page configuration
st.set_page_config(
    page_title="Panel de Mensajes",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main app logic
if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    login()
else:
    display_dashboard()

def get_next_id(df):
    # Convert all ids to integers, ignoring non-numeric ids
    numeric_ids = pd.to_numeric(df['id'], errors='coerce')
    # Get the maximum numeric id, defaulting to 0 if none found
    max_id = numeric_ids.max()
    if pd.isna(max_id):
        return '1'
    return str(int(max_id) + 1)
