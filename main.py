from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from routes import router

app = FastAPI(
    title="SmolLM RAG API",
    description="An API that performs RAG using SmolLM-135M",
    version="1.0.0"
)

# Include your router with all endpoints (/upsert/git, /upload, /query, etc.)
app.include_router(router)

# Custom handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid input format",
            "details": exc.errors()
        },
    )

# Optional: root route
@app.get("/")
async def root():
    return {"message": "Welcome to the SmolLM RAG API"}
