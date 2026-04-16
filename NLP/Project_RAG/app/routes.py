from fastapi import APIRouter, HTTPException, UploadFile, File
from app.service import RAGService
from pydantic import BaseModel
from app.config import Settings


router = APIRouter()

rag_service = RAGService(api_key=Settings.OPENAI_API_KEY)

class AskRequest(BaseModel):
    query: str


@router.get("/health")
def health():
    return {'status': 'ok'}


@router.post('/ask')
def ask(request: AskRequest):
    try:
        result = rag_service.ask(request.query)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/rebuild")
def rebuild():
    try:
        result = rag_service.rebuild_index()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
def upload(file: UploadFile = File(...)):
    contents = file.file.read()

    with open(f'data/raw/{file.filename}', 'wb') as f:
        f.write(contents)
    return {'message': f'{file.filename} uploaded successfully'}