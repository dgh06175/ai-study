"""
SSE Practice Project - Web Frontend
Flask를 사용한 웹 기반 SSE 클라이언트
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import requests
import json
from typing import Optional

app = Flask(__name__)

# 백엔드 API 설정
BACKEND_URL = "http://localhost:8000"


class BackendClient:
    """백엔드 API 클라이언트"""

    def __init__(self):
        self.session = requests.Session()

    def create_room(self, name: Optional[str] = None) -> dict:
        """대화방 생성"""
        data = {"name": name} if name else {}
        response = self.session.post(f"{BACKEND_URL}/rooms", json=data)
        response.raise_for_status()
        return response.json()

    def list_rooms(self) -> dict:
        """대화방 목록 조회"""
        response = self.session.get(f"{BACKEND_URL}/rooms")
        response.raise_for_status()
        return response.json()

    def update_room(self, room_id: str, name: str) -> dict:
        """대화방 이름 변경"""
        data = {"name": name}
        response = self.session.put(f"{BACKEND_URL}/rooms/{room_id}", json=data)
        response.raise_for_status()
        return response.json()

    def delete_room(self, room_id: str) -> dict:
        """대화방 삭제"""
        response = self.session.delete(f"{BACKEND_URL}/rooms/{room_id}")
        response.raise_for_status()
        return response.json()


# 글로벌 클라이언트 인스턴스
backend_client = BackendClient()


@app.route("/")
def index():
    """메인 페이지"""
    try:
        rooms_data = backend_client.list_rooms()
        rooms = rooms_data.get("rooms", [])
        return render_template("index.html", rooms=rooms)
    except requests.RequestException as e:
        return render_template(
            "index.html", rooms=[], error=f"백엔드 서버 연결 오류: {e}"
        )


@app.route("/rooms", methods=["POST"])
def create_room():
    """대화방 생성"""
    try:
        room_name = request.form.get("room_name", "").strip()
        if not room_name:
            room_name = None

        room = backend_client.create_room(room_name)
        return redirect(url_for("chat_room", room_id=room["id"]))
    except requests.RequestException as e:
        return jsonify({"error": f"대화방 생성 실패: {e}"}), 500


@app.route("/rooms/<room_id>")
def chat_room(room_id):
    """채팅방 페이지"""
    try:
        rooms_data = backend_client.list_rooms()
        rooms = rooms_data.get("rooms", [])

        # 현재 방 정보 찾기
        current_room = None
        for room in rooms:
            if room["id"] == room_id:
                current_room = room
                break

        if not current_room:
            return redirect(url_for("index"))

        return render_template("chat.html", room=current_room, rooms=rooms)
    except requests.RequestException as e:
        return render_template(
            "chat.html", room=None, rooms=[], error=f"백엔드 서버 연결 오류: {e}"
        )


@app.route("/rooms/<room_id>/delete", methods=["POST"])
def delete_room(room_id):
    """대화방 삭제"""
    try:
        backend_client.delete_room(room_id)
        return redirect(url_for("index"))
    except requests.RequestException as e:
        return jsonify({"error": f"대화방 삭제 실패: {e}"}), 500


@app.route("/api/rooms")
def api_rooms():
    """대화방 목록 API"""
    try:
        rooms_data = backend_client.list_rooms()
        return jsonify(rooms_data)
    except requests.RequestException as e:
        return jsonify({"error": f"대화방 목록 조회 실패: {e}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
