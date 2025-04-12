import os

class Config:
    # Database configuration
   
    SESSION_TYPE = 'filesystem'
    SECRET_KEY = os.getenv('SECRET_KEY', 'default_secret_key')

    # API Keys and other configurations with defaults
    OPENAI_API_KEY = "your-openai-api-key"
    
    # Twilio configuration
    TWILIO_ACCOUNT_SID = "your-twilio-account-sid"
    TWILIO_AUTH_TOKEN = "your-twilio-auth-token"
    
    # Path to the PDF file for knowledge base
    PDF_FILE_PATH = "Mercy_Update.pdf"
    
    # Optional: AWS configuration if you want to use DynamoDB for session storage
    # AWS_REGION = "your-aws-region"
    # DYNAMODB_TABLE_NAME = "your-dynamodb-table-name"
    