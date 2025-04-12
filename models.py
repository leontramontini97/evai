from datetime import datetime, timezone, timedelta
import logging
from openai import OpenAI
import boto3
import os
import re
import hashlib
import uuid
from dotenv import load_dotenv

# Load environment variables

####WARNINNG the version deployed on ec2 is slightly different


dynamodb = boto3.resource("dynamodb", region_name="us-east-2")

table = dynamodb.Table('chaatbotmercy')

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
categories_dict = {
    1: "Ginecología Clínica",
    2: "Cirugía Ginecológica",
    3: "Ginecología Estética",
    4: "Sexualidad",
    5: "Salud y Bienestar",
    6: "Información sobre agendamiento de citas, seguros, precios y demás",
    7: "Saludos y otros",
}

def anonymize_phone_number(phone_number):
    """
    Create a deterministic UUID from a phone number to maintain session continuity
    while providing anonymization.
    """
    # Generate a UUID v5 using the phone number as the name
    # UUID namespace is a fixed UUID (using the NAMESPACE_DNS here as a standard namespace)
    # This ensures the same phone number always generates the same UUID
    namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # NAMESPACE_DNS
    return str(uuid.uuid5(namespace, str(phone_number)))

def classify_question(question):
    # This is for chat models like 'gpt-4o-mini', 'gpt-3.5-turbo', etc.
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """
                Tu tarea es clasificar preguntas hechas a un chatbot de una ginecóloga en estas categorías:
                1: Ginecología Clínica
                2: Cirugía Ginecológica
                3: Ginecología Estética
                4: Sexualidad
                5: Salud y Bienestar
                6: Información sobre agendamiento de citas, seguros, precios y demás
                7: Saludos y otros

                IMPORTANTE: Responde ÚNICAMENTE con el número de la categoría (1-7). No escribas nada más!!!! Esto es muy importante.
                """},
                {"role": "user", "content": question},
            ]
        )
        # Access the content attribute directly from the message object
        response = completion.choices[0].message.content.strip()
        try:
            # Try to convert the response directly to an integer
            category_number = int(response)
            if 1 <= category_number <= 7:
                return categories_dict[category_number]
        except ValueError:
            # If direct conversion fails, try to extract number using regex
            number_match = re.search(r'\d+', response)
            if number_match:
                category_number = int(number_match.group())
                if 1 <= category_number <= 7:
                    return categories_dict[category_number]
        
        print(f"Invalid response for question: {question} -> Response: {response}")
        return "Error"
    except Exception as e:
        print(f"Error processing question: {question} -> {e}")
        return "Error"


def log_conversation(session_id, question, answer):
    """Log a conversation to DynamoDB"""
    try:
        # Anonymize the session_id (phone number) for storage while maintaining session tracking
        anonymized_session_id = anonymize_phone_number(session_id)
        
        # Get category for the question
        category = classify_question(question)
        
        # Generate timestamp in format matching existing records
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Get the highest existing ID and increment, starting from 254 if empty
        try:
            response = table.scan(
                ProjectionExpression='id',
                Select='SPECIFIC_ATTRIBUTES'
            )
            existing_ids = [int(item['id']) for item in response['Items']]
            next_id = str(max(max(existing_ids), 254) + 1) if existing_ids else '254'
        except Exception as e:
            logging.error(f"Error getting next ID: {e}")
            next_id = '254'
        
        # Create item for DynamoDB
        item = {
            'id': next_id,
            'session_id': anonymized_session_id,  # Store anonymized session_id
            'message_body': question,
            'response': answer,
            'category': category,
            'timestamp': timestamp
        }
        
        # Put item in DynamoDB table
        table.put_item(Item=item)
        
        logging.info(f"Conversation logged successfully with ID: {next_id}")
        
    except Exception as e:
        logging.error(f"Failed to log conversation: {str(e)}")
