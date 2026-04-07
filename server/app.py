import uvicorn
from fastapi import FastAPI
from productivity_env import ProductivityEnv

app = FastAPI()
env = ProductivityEnv()

@app.get("/health")
def health():
    return {"status": "ok"}

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
