from fastapi import FastAPI
from app.controllers import empresa_controller

app = FastAPI()

app.include_router(empresa_controller.router)
