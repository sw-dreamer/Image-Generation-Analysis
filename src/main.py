import nest_asyncio
import asyncio
import uvicorn
from pyngrok import ngrok
from .config import NGROK_AUTH_TOKEN
from .web_app import app

ngrok.set_auth_token(NGROK_AUTH_TOKEN)
public_url = ngrok.connect(8000)
print("🌍 Public URL:", public_url)

if __name__ == "__main__":
    nest_asyncio.apply()
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, reload=True, log_level="debug")
    server = uvicorn.Server(config)
    asyncio.get_event_loop().run_until_complete(server.serve())

