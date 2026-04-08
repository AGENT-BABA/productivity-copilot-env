import os

import uvicorn
from openenv.core import create_fastapi_app

from productivity_env import ProductivityAction, ProductivityEnv, ProductivityObservation


def make_env() -> ProductivityEnv:
    return ProductivityEnv(task_name=os.getenv("PRODUCTIVITY_TASK", "triage"))


app = create_fastapi_app(
    env=make_env,
    action_cls=ProductivityAction,
    observation_cls=ProductivityObservation,
)

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=int(os.getenv("PORT", "7860")))

if __name__ == "__main__":
    main()
