import os
import sys
from app import app

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 5000))
DEBUG = os.getenv("DEBUG", "False") == "True"

if __name__ == "__main__":
    try:
        app.run(host=HOST, port=PORT, debug=DEBUG)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
