# 예시: 기존 베이스 이미지 아래에 추가합니다.
FROM python:3.12-slim 

# 💡 [★핵심 추가] LightGBM 멀티스레딩 구동에 필수인 libgomp1 라이브러리 설치
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- 아래는 기존에 작성해두신 Dockerfile 내용들을 그대로 유지하시면 됩니다 ---
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]