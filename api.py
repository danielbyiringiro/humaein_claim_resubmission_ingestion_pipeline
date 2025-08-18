import os
import tempfile
from typing import List, Dict
from main import ClaimsProcessor
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, UploadFile, File

app = FastAPI(title = "Claims Processor Pipeline")

processor = ClaimsProcessor()

@app.post('/upload', response_model = List[dict])
async def upload_file(file: UploadFile = File(...)):
    # Validate file type

    file_ext = os.path.splitext(file.filename)[1].lower()
    print(file_ext)
    
    if file_ext not in ('.csv', '.json'):
        raise HTTPException(
            status_code = 400,
            detail = 'Only CSV and JSON files are allowed'
        )

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp:
            content = await file.read()
            temp.write(content)
            tmp_path = temp.name
            print(tmp_path)
        
        # Process the file
        eligible_claims = processor.process_file(tmp_path)

        # Clean up
        os.unlink(tmp_path)

        return eligible_claims

    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail = f"Error processing file, {str(e)}"
        )

@app.get('/rejected', response_model = List[dict])
async def get_rejected_claims():
    return processor.rejected_claims

@app.get('/metrics', response_model = Dict)
async def get_metrics():
    return processor.metrics






