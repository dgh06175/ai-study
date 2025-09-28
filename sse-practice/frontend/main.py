"""
SSE Practice Project - Frontend GUI Client
tkinter를 사용한 간단한 GUI 클라이언트
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import requests
import json
import threading
import time
from typing import Optional
import hashlib
import uuid


class SSEClient:
    """SSE 클라이언트 클래스"""

    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.session = requests.Session()

    def create_room(self, name: Optional[str] = None) -> dict:
        """대화방 생성"""
        data = {"name": name} if name else {}
        response = self.session.post(f"{self.base_url}/rooms", json=data)
        response.raise_for_status()
        return response.json()

    def list_rooms(self) -> dict:
        """대화방 목록 조회"""
        response = self.session.get(f"{self.base_url}/rooms")
        response.raise_for_status()
        return response.json()

    def update_room(self, room_id: str, name: str) -> dict:
        """대화방 이름 변경"""
        data = {"name": name}
        response = self.session.put(f"{self.base_url}/rooms/{room_id}", json=data)
        response.raise_for_status()
        return response.json()

    def delete_room(self, room_id: str) -> dict:
        """대화방 삭제"""
        response = self.session.delete(f"{self.base_url}/rooms/{room_id}")
        response.raise_for_status()
        return response.json()

    def chat_with_sse(self, room_id: str, message: str, callback):
        """SSE를 통한 채팅"""
        data = {"message": message}
        url = f"{self.base_url}/rooms/{room_id}/chat"

        response = self.session.post(
            url, json=data, stream=True, headers={"Accept": "text/event-stream"}
        )
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                try:
                    event_data = json.loads(line[6:])  # "data: " 제거
                    callback(event_data)
                    if event_data.get("finished", False):
                        break
                except json.JSONDecodeError:
                    continue


class ChatGUI:
    """채팅 GUI 메인 클래스"""

    def __init__(self):
        self.sse_client = SSEClient()
        self.current_room_id = None
        self.is_streaming = False

        # GUI 초기화
        self.setup_gui()
        self.refresh_rooms()

    def setup_gui(self):
        """GUI 구성"""
        self.root = tk.Tk()
        self.root.title("SSE 채팅 클라이언트")
        self.root.geometry("800x600")

        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        # 대화방 관리 섹션
        room_frame = ttk.LabelFrame(main_frame, text="대화방 관리", padding="10")
        room_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        # 새 대화방 생성
        ttk.Label(room_frame, text="새 대화방 이름:").grid(row=0, column=0, sticky=tk.W)
        self.new_room_name = tk.StringVar()
        new_room_entry = ttk.Entry(
            room_frame, textvariable=self.new_room_name, width=30
        )
        new_room_entry.grid(row=0, column=1, padx=(5, 5))
        ttk.Button(room_frame, text="생성", command=self.create_room).grid(
            row=0, column=2
        )

        # 대화방 목록
        ttk.Label(room_frame, text="대화방 목록:").grid(
            row=1, column=0, sticky=tk.W, pady=(10, 0)
        )

        # 대화방 선택 콤보박스
        self.room_var = tk.StringVar()
        self.room_combo = ttk.Combobox(
            room_frame, textvariable=self.room_var, width=40, state="readonly"
        )
        self.room_combo.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.room_combo.bind("<<ComboboxSelected>>", self.on_room_selected)

        # 대화방 관리 버튼들
        button_frame = ttk.Frame(room_frame)
        button_frame.grid(row=2, column=2, padx=(5, 0))
        ttk.Button(button_frame, text="새로고침", command=self.refresh_rooms).pack(
            side=tk.LEFT, padx=(0, 5)
        )
        ttk.Button(button_frame, text="삭제", command=self.delete_room).pack(
            side=tk.LEFT
        )

        # 채팅 섹션
        chat_frame = ttk.LabelFrame(main_frame, text="채팅", padding="10")
        chat_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, 10))

        # 현재 대화방 표시
        self.current_room_label = ttk.Label(chat_frame, text="대화방을 선택해주세요")
        self.current_room_label.grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10)
        )

        # 채팅 내역 표시
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            width=80,
            height=20,
            state=tk.DISABLED,
            wrap=tk.WORD,
            bg="black",
            fg="white",
        )
        self.chat_display.grid(
            row=1, column=0, columnspan=2, sticky="nsew", pady=(0, 10)
        )

        # 메시지 입력
        message_frame = ttk.Frame(chat_frame)
        message_frame.grid(row=2, column=0, columnspan=2, sticky="ew")

        ttk.Label(message_frame, text="메시지:").grid(row=0, column=0, sticky="w")
        self.message_var = tk.StringVar()
        self.message_entry = ttk.Entry(
            message_frame, textvariable=self.message_var, width=60
        )
        self.message_entry.grid(row=0, column=1, padx=(5, 5), sticky="ew")
        self.message_entry.bind("<Return>", lambda e: self.send_message())

        self.send_button = ttk.Button(
            message_frame, text="전송", command=self.send_message
        )
        self.send_button.grid(row=0, column=2)

        # 그리드 설정
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        room_frame.columnconfigure(1, weight=1)
        chat_frame.columnconfigure(1, weight=1)
        chat_frame.rowconfigure(1, weight=1)
        message_frame.columnconfigure(1, weight=1)

    def refresh_rooms(self):
        """대화방 목록 새로고침"""
        try:
            rooms_data = self.sse_client.list_rooms()
            rooms = rooms_data.get("rooms", [])

            room_options = [f"{room['name']} ({room['id']})" for room in rooms]
            self.room_combo["values"] = room_options

            if rooms:
                self.room_combo.set(room_options[0])
                self.on_room_selected()
        except requests.RequestException as e:
            messagebox.showerror("오류", f"대화방 목록을 불러올 수 없습니다: {e}")

    def create_room(self):
        """새 대화방 생성"""
        room_name = self.new_room_name.get().strip()
        if not room_name:
            room_name = None  # 서버에서 자동 생성

        try:
            room = self.sse_client.create_room(room_name)
            self.new_room_name.set("")
            self.refresh_rooms()
            messagebox.showinfo("성공", f"대화방 '{room['name']}'이 생성되었습니다.")
        except requests.RequestException as e:
            messagebox.showerror("오류", f"대화방을 생성할 수 없습니다: {e}")

    def delete_room(self):
        """현재 선택된 대화방 삭제"""
        if not self.current_room_id:
            messagebox.showwarning("경고", "삭제할 대화방을 선택해주세요.")
            return

        if messagebox.askyesno("확인", "정말로 이 대화방을 삭제하시겠습니까?"):
            try:
                self.sse_client.delete_room(self.current_room_id)
                self.current_room_id = None
                self.current_room_label.config(text="대화방을 선택해주세요")
                self.clear_chat_display()
                self.refresh_rooms()
                messagebox.showinfo("성공", "대화방이 삭제되었습니다.")
            except requests.RequestException as e:
                messagebox.showerror("오류", f"대화방을 삭제할 수 없습니다: {e}")

    def on_room_selected(self, event=None):
        """대화방 선택 시 호출"""
        selected = self.room_var.get()
        if selected:
            # 선택된 항목에서 room_id 추출 (괄호 안의 ID)
            try:
                room_id = selected.split("(")[-1].split(")")[0]
                self.current_room_id = room_id
                room_name = selected.split(" (")[0]
                self.current_room_label.config(text=f"현재 대화방: {room_name}")
                self.clear_chat_display()
            except IndexError:
                pass

    def clear_chat_display(self):
        """채팅 내역 초기화"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def append_to_chat(self, text: str, tag: Optional[str] = None):
        """채팅 내역에 텍스트 추가"""
        self.chat_display.config(state=tk.NORMAL)
        if tag:
            self.chat_display.insert(tk.END, text, tag)
        else:
            self.chat_display.insert(tk.END, text)
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def send_message(self):
        """메시지 전송"""
        if not self.current_room_id:
            messagebox.showwarning("경고", "대화방을 선택해주세요.")
            return

        message = self.message_var.get().strip()
        if not message:
            messagebox.showwarning("경고", "메시지를 입력해주세요.")
            return

        if self.is_streaming:
            messagebox.showwarning("경고", "이전 메시지의 응답을 기다리고 있습니다.")
            return

        # 사용자 메시지 표시
        self.append_to_chat(f"\n>>> 사용자: {message}\n", "user")
        self.append_to_chat("🤖 AI: ", "ai_label")

        # 메시지 입력창 비우기 및 버튼 비활성화
        self.message_var.set("")
        self.send_button.config(state=tk.DISABLED)
        self.is_streaming = True

        # SSE 스트리밍을 별도 스레드에서 실행
        threading.Thread(
            target=self.stream_response,
            args=(self.current_room_id, message),
            daemon=True,
        ).start()

    def stream_response(self, room_id: str, message: str):
        """SSE 응답 스트리밍 처리"""
        try:

            def on_token(event_data):
                token = event_data.get("token", "")
                finished = event_data.get("finished", False)

                if token:
                    # GUI 업데이트는 메인 스레드에서 실행
                    self.root.after(
                        0, lambda: self.append_to_chat(f"{token} ", "ai_response")
                    )

                if finished:
                    final_message = event_data.get("message", "")
                    self.root.after(
                        0,
                        lambda: self.append_to_chat(
                            f"\n\n✅ {final_message}\n", "finished"
                        ),
                    )

            self.sse_client.chat_with_sse(room_id, message, on_token)

        except requests.RequestException as e:
            error_msg = f"\n❌ 오류 발생: {e}\n"
            self.root.after(0, lambda: self.append_to_chat(error_msg, "error"))
        finally:
            # 스트리밍 완료 후 버튼 활성화
            self.root.after(0, self.on_streaming_finished)

    def on_streaming_finished(self):
        """스트리밍 완료 후 처리"""
        self.is_streaming = False
        self.send_button.config(state=tk.NORMAL)

    def setup_text_tags(self):
        """텍스트 태그 설정"""
        self.chat_display.tag_config(
            "user", foreground="blue", font=("Arial", 10, "bold")
        )
        self.chat_display.tag_config(
            "ai_label", foreground="green", font=("Arial", 10, "bold")
        )
        self.chat_display.tag_config("ai_response", foreground="white")
        self.chat_display.tag_config(
            "finished", foreground="gray", font=("Arial", 9, "italic")
        )
        self.chat_display.tag_config(
            "error", foreground="red", font=("Arial", 10, "bold")
        )

    def run(self):
        """GUI 실행"""
        self.setup_text_tags()
        self.root.mainloop()


def main():
    """메인 함수"""
    try:
        app = ChatGUI()
        app.run()
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
    except Exception as e:
        print(f"오류 발생: {e}")


if __name__ == "__main__":
    main()
