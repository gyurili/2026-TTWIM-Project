from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 모든 도메인에서의 접속을 허용하는 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포시에는 API Gateway 주소만 넣는 것이 좋습니다.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Hello Azas!"}