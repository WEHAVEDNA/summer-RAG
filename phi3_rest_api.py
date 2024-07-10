from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from phi3_model import Phi3
from fastapi.responses import StreamingResponse

class Message(BaseModel):
    role: str
    content: str

class GenerateRequest(BaseModel):
    messages: List[Message]
    model: str = 'cuda/cuda-int4-rtn-block-32'
    min_length: Optional[int] = None
    max_length: int = 4096
    do_sample: bool = False
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    temperature: Optional[float] = None
    repetition_penalty: Optional[float] = None

app = FastAPI()

@app.post("/generate/")
async def generate_text(request: GenerateRequest):
    try:
        input_text = "\n".join([msg.content for msg in request.messages])
        
        phi3_instance = Phi3(
            input_text=input_text,
            model=request.model,
            min_length=request.min_length,
            max_length=request.max_length,
            do_sample=request.do_sample,
            top_p=request.top_p,
            top_k=request.top_k,
            temperature=request.temperature,
            repetition_penalty=request.repetition_penalty
        )
        
        def token_stream():
            for token in phi3_instance.generate_tokens():
                yield token
        
        return StreamingResponse(token_stream(), media_type="text/plain")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# fastapi run phi3_rest_api.py