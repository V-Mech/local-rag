from __future__ import annotations

import asyncio
import logging
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.agent_manager import AgentManager

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
    project_directory = Path(__file__).resolve().parent.parent
    (project_directory / "uploads").mkdir(exist_ok=True)
    (project_directory / "data").mkdir(exist_ok=True)
    app.state.agent_manager = AgentManager(project_directory)
    app.state.project_directory = project_directory
    yield

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self' 'unsafe-inline'"
    return response

def _session_values(request: Request) -> tuple[str, str, bool]:
    session_id = request.cookies.get("rag_session")
    csrf_token = request.cookies.get("rag_csrf")
    is_new = not session_id or not csrf_token
    return session_id or str(uuid.uuid4()), csrf_token or secrets.token_urlsafe(32), is_new

def _set_session_cookies(response: HTMLResponse, session_id: str, csrf_token: str) -> None:
    secure = os.getenv("COOKIE_SECURE", "false").lower() == "true"
    response.set_cookie("rag_session", session_id, httponly=True, secure=secure, samesite="lax")
    response.set_cookie("rag_csrf", csrf_token, httponly=False, secure=secure, samesite="lax")

def _csrf_is_valid(request: Request, csrf_token: str) -> bool:
    cookie_token = request.cookies.get("rag_csrf")
    return bool(cookie_token and secrets.compare_digest(cookie_token, csrf_token))

async def _template_context(request: Request, **kwargs: object) -> dict[str, object]:
    manager: AgentManager = request.app.state.agent_manager
    context: dict[str, object] = {
        "request": request,
        "documents": await asyncio.to_thread(manager.list_documents),
        "csrf_token": request.cookies.get("rag_csrf", ""),
        "current_document": "",
        "current_document_id": "",
    }
    context.update(kwargs)
    return context

@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    session_id, csrf_token, _ = _session_values(request)
    response = templates.TemplateResponse(request, "index.html", await _template_context(request, csrf_token=csrf_token))
    _set_session_cookies(response, session_id, csrf_token)
    return response

async def _save_validated_upload(file: UploadFile, destination: Path, max_bytes: int) -> None:
    allowed_extensions = {".pdf", ".docx", ".pptx", ".xml", ".txt"}
    if destination.suffix.lower() not in allowed_extensions:
        raise ValueError("Supported file types are PDF, DOCX, PPTX, XML, and TXT")

    size = 0
    with destination.open("wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise ValueError("The upload exceeds the 25 MB limit")
            buffer.write(chunk)

    with destination.open("rb") as uploaded_file:
        header = uploaded_file.read(8)
    extension = destination.suffix.lower()
    if extension == ".pdf" and not header.startswith(b"%PDF-"):
        raise ValueError("The uploaded file is not a valid PDF")
    if extension in {".docx", ".pptx"} and not header.startswith(b"PK"):
        raise ValueError("The uploaded Office file is not a valid ZIP package")

@app.post("/upload", response_class=HTMLResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    csrf_token: str = Form(...),
) -> HTMLResponse:
    if not _csrf_is_valid(request, csrf_token):
        return templates.TemplateResponse(
            request, "index.html", await _template_context(request, error="Your session expired. Refresh and try again."), status_code=403
        )

    original_name = Path((file.filename or "").replace("\\", "/")).name
    if not original_name or original_name in {".", ".."}:
        return templates.TemplateResponse(
            request, "index.html", await _template_context(request, error="A valid filename is required."), status_code=400
        )

    project_directory: Path = request.app.state.project_directory
    destination = project_directory / "uploads" / f"{uuid.uuid4()}{Path(original_name).suffix.lower()}"
    try:
        await _save_validated_upload(file, destination, max_bytes=25 * 1024 * 1024)
        manager: AgentManager = request.app.state.agent_manager
        indexed = await asyncio.to_thread(manager.ingest_document, destination, original_name)
    except Exception as error:

        logging.getLogger(__name__).warning("Upload failed for %s: %s", original_name, error)
        destination.unlink(missing_ok=True)
        return templates.TemplateResponse(
            request, "index.html", await _template_context(request, error="The document could not be indexed. Check its type and content."), status_code=400
        )
    finally:
        await file.close()

    session_id, csrf_cookie, _ = _session_values(request)
    response = templates.TemplateResponse(
        request,
        "index.html",
        await _template_context(
            request,
            message=f"{original_name} uploaded successfully.",
            chunks=indexed["chunks"],
            selected_document=indexed["document_id"],
            current_document=indexed["source"],
            current_document_id=indexed["document_id"],
            csrf_token=csrf_cookie,
        ),
    )
    _set_session_cookies(response, session_id, csrf_cookie)
    return response

@app.post("/ask", response_class=HTMLResponse)
async def ask_question(
    request: Request,
    question: str = Form(..., min_length=1, max_length=4_000),
    document: str = Form(""),
    csrf_token: str = Form(...),
) -> HTMLResponse:
    if not _csrf_is_valid(request, csrf_token):
        return templates.TemplateResponse(
            request, "index.html", await _template_context(request, error="Your session expired. Refresh and try again."), status_code=403
        )

    manager: AgentManager = request.app.state.agent_manager
    documents = await asyncio.to_thread(manager.list_documents)
    selected_document = document.strip()
    valid_ids = {entry["document_id"] for entry in documents}
    if selected_document and selected_document not in valid_ids:
        return templates.TemplateResponse(
            request, "index.html", await _template_context(request, error="The selected document no longer exists."), status_code=400
        )

    session_id, csrf_cookie, _ = _session_values(request)
    result = await asyncio.to_thread(manager.ask, question.strip(), session_id, selected_document or None)
    response = templates.TemplateResponse(
        request,
        "index.html",
        await _template_context(
            request,
            question=question,
            answer=result["answer"],
            plan=result["plan"]["steps"],
            confidence=result.get("confidence", 0),
            hallucination_risk=result.get("verification", {}).get("hallucination_risk", "high"),
            selected_document=selected_document,
            csrf_token=csrf_cookie,
        ),
    )
    _set_session_cookies(response, session_id, csrf_cookie)
    return response
