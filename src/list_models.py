from google import genai
import os
from dotenv import load_dotenv

load_dotenv()  # this reads .env from the project root

def main():
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for model in client.models.list():
        print(model.name)

if __name__ == "__main__":
    main()