"""
SSE Practice Project - Backend Server
간단한 LLM 답변처럼 문자열을 천천히 반환하는 FastAPI 서버
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import json
import hashlib
import sqlite3
import os
from typing import Optional
import uuid

app = FastAPI(
    title="SSE Practice API",
    description="LLM처럼 문자열을 천천히 반환하는 SSE 서버",
    version="1.0.0",
)

# 데이터베이스 초기화
DATABASE_PATH = "chat_rooms.db"


def init_db():
    """데이터베이스 초기화"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_rooms (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.commit()
    conn.close()


# 앱 시작 시 데이터베이스 초기화
init_db()


# 요청/응답 모델
class ChatRoomCreate(BaseModel):
    name: Optional[str] = None


class ChatRoomUpdate(BaseModel):
    name: str


class ChatMessage(BaseModel):
    message: str


class ChatRoom(BaseModel):
    id: str
    name: str
    created_at: str


@app.get("/")
async def root():
    """API 상태 확인"""
    return {"message": "SSE Practice API Server is running!"}


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy"}


# 데이터베이스 헬퍼 함수들
def generate_room_id(name: str = None) -> str:
    """대화방 ID를 해시값으로 생성"""
    if name:
        return hashlib.md5(name.encode()).hexdigest()[:8]
    else:
        return str(uuid.uuid4())[:8]


def get_db_connection():
    """데이터베이스 연결 반환"""
    return sqlite3.connect(DATABASE_PATH)


# 대화방 관련 API 엔드포인트
@app.post("/rooms", response_model=ChatRoom)
async def create_chat_room(room_data: ChatRoomCreate):
    """대화방 생성"""
    room_name = room_data.name or f"Room_{generate_room_id()}"
    room_id = generate_room_id(room_name)

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO chat_rooms (id, name) VALUES (?, ?)", (room_id, room_name)
        )
        conn.commit()

        # 생성된 방 정보 조회
        cursor.execute(
            "SELECT id, name, created_at FROM chat_rooms WHERE id = ?", (room_id,)
        )
        result = cursor.fetchone()

        return ChatRoom(id=result[0], name=result[1], created_at=result[2])
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Room already exists")
    finally:
        conn.close()


@app.get("/rooms")
async def list_chat_rooms():
    """대화방 목록 조회"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "SELECT id, name, created_at FROM chat_rooms ORDER BY created_at DESC"
        )
        results = cursor.fetchall()

        rooms = [ChatRoom(id=row[0], name=row[1], created_at=row[2]) for row in results]
        return {"rooms": rooms}
    finally:
        conn.close()


@app.put("/rooms/{room_id}", response_model=ChatRoom)
async def update_chat_room(room_id: str, room_data: ChatRoomUpdate):
    """대화방 이름 변경"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "UPDATE chat_rooms SET name = ? WHERE id = ?", (room_data.name, room_id)
        )

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Room not found")

        conn.commit()

        # 업데이트된 방 정보 조회
        cursor.execute(
            "SELECT id, name, created_at FROM chat_rooms WHERE id = ?", (room_id,)
        )
        result = cursor.fetchone()

        return ChatRoom(id=result[0], name=result[1], created_at=result[2])
    finally:
        conn.close()


@app.delete("/rooms/{room_id}")
async def delete_chat_room(room_id: str):
    """대화방 삭제"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("DELETE FROM chat_rooms WHERE id = ?", (room_id,))

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Room not found")

        conn.commit()
        return {"message": "Room deleted successfully"}
    finally:
        conn.close()


# SSE 관련 함수들
async def generate_sse_response(message: str):
    """LLM처럼 천천히 응답을 생성하는 제너레이터"""
    # 사용자 메시지를 10번 반복하는 응답 생성
    response_text = f"안녕하세요! 당신의 메시지 '{message}'에 대한 답변입니다. " * 10

    # 토큰 단위로 분할 (공백 기준)
    tokens = response_text.split()

    # 초당 30토큰 속도로 전송 (1초 / 30토큰 = ~0.033초)
    delay_per_token = 1.0 / 30

    for token in tokens:
        # SSE 형식으로 데이터 전송
        data = {"token": token, "finished": False}
        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        await asyncio.sleep(delay_per_token)

    # 완료 신호 전송
    final_data = {"token": "", "finished": True, "message": "응답이 완료되었습니다."}
    yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"


@app.post("/rooms/{room_id}/chat")
async def chat_with_sse(room_id: str, message: ChatMessage):
    """SSE를 통한 채팅 응답"""
    # 방 존재 여부 확인
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT id FROM chat_rooms WHERE id = ?", (room_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Room not found")
    finally:
        conn.close()

    # SSE 스트리밍 응답 반환
    return StreamingResponse(
        generate_sse_response(message.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )


# CORS 미들웨어 추가
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서버 실행을 위한 메인 함수
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
