from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.user import router as user_router

app = FastAPI(title="azas-backend API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_router, prefix="/user")

@app.get("/")
async def root():
    return {"message": "AZAS API"}
