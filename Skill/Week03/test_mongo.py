
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv(override=True)

uri = os.getenv("MONGODB_URI")

try:
    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=5000
    )

    client.admin.command("ping")

    print("MongoDB Connected Successfully!")

    db = client["llm_monitoring"]
    collection = db["api_usage"]

    collection.insert_one({
        "message": "Week 3 connection test",
        "status": "success"
    })

    print("Test record inserted successfully!")

    client.close()

except Exception as error:
    print("Connection failed:", error)
