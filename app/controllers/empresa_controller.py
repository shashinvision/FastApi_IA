from fastapi import APIRouter
from app.services.empresa_service import EmpresaService
from app.models.empresa_model import QuestionRequest

router = APIRouter()
service = EmpresaService()


@router.post("/empresa/ask")
async def ask_empresa_question(request: QuestionRequest):
    response = service.ask(request.question)
    return {"response": response}
