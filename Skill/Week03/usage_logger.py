
import os
import time
from datetime import datetime, timezone

from pymongo import MongoClient
from pymongo.errors import PyMongoError


class UsageLogger:

    def __init__(self):

        mongo_uri = os.getenv("MONGODB_URI")

        if not mongo_uri:
            raise ValueError("MONGODB_URI not found in .env")

        self.client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=10000,
            retryWrites=True
        )

        self.client.admin.command("ping")

        db = self.client["llm_monitoring"]
        self.collection = db["api_usage"]

        print("MongoDB connected successfully!")

    def log_call(self, record):

        document = dict(record)
        document["timestamp"] = datetime.now(timezone.utc)

        for attempt in range(3):

            try:
                self.collection.insert_one(document)
                print("Usage details saved to MongoDB.")
                return True

            except PyMongoError as error:

                print(f"MongoDB write failed: {error}")

                if attempt < 2:
                    time.sleep(2 ** attempt)

        return False

    def close(self):
        self.client.close()
