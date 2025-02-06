from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from document_query_app import DocumentQueryApp

app = FastAPI()

# Enable CORS for local development (adjust allowed origins in production)
app.add_middleware(
    CORSMiddleware,
    # In production, restrict this to your front-end's domain.
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize your DocumentQueryApp once at startup.
query_app = DocumentQueryApp()


# Define the expected JSON payload.
class QueryRequest(BaseModel):
    query: str
    additional_context: str = ""
    model: str = ""  # Accept a comma-separated string of model names if needed


@app.get("/models")
def get_models():
    # Fetch models from the DocumentQueryApp instance.
    # Access the available models from the instance
    models = query_app.available_models
    return {"models": [{"value": model, "name": model} for model in models]}


@app.post("/query")
def query_endpoint(request: QueryRequest):
    try:
        # Call the process_query method to get the result.
        result = query_app.process_query(
            request.query, request.additional_context, request.model)
        # Return the response with an explicit charset=utf-8 header.
        return JSONResponse(content={"result": result}, headers={"Content-Type": "application/json; charset=utf-8"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the API server on localhost port 8000.
    uvicorn.run(app, host="127.0.0.1", port=8000)
