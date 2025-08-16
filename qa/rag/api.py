import os
import json
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from crew import run_crew
import uvicorn

from common.settings import settings

app = FastAPI()

class UserRequest(BaseModel):
    message: str
    user_id: str

SESSIONS_DIR = getattr(settings, "SESSIONS_DIR", os.path.join(os.getcwd(), "sessions"))
os.makedirs(SESSIONS_DIR, exist_ok=True)

MAX_HISTORY_CHARS = getattr(settings, "MAX_SESSION_HISTORY_CHARS", 18_000)


def _session_path(user_id: str) -> str:
    safe = "".join([c if c.isalnum() or c in "._-" else "_" for c in user_id])
    return os.path.join(SESSIONS_DIR, f"{safe}.json")

def _load_history(user_id: str):
    path = _session_path(user_id)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _save_history(user_id: str, history):
    path = _session_path(user_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def _append_turn(user_id: str, role: str, text: str):
    history = _load_history(user_id)
    history.append({"role": role, "text": text, "ts": time.time()})
    # Trimming: if the total length exceeds the limit, we remove the old moves.
    joined = "\n".join([f"{h['role']}: {h['text']}" for h in history])
    if len(joined) > MAX_HISTORY_CHARS:
        # We will save the last moves until we finish.
        new_hist = []
        rev = list(reversed(history))
        curr = ""
        for h in rev:
            candidate = f"{h['role']}: {h['text']}\n" + curr
            if len(candidate) > MAX_HISTORY_CHARS:
                break
            curr = candidate
            new_hist.insert(0, h)  # rebuild in the correct order
        if new_hist:
            history = new_hist
        else:
            # if one move > limit — cut it off
            last = history[-1]
            last_text = last["text"][-(MAX_HISTORY_CHARS//2):]  # leave a tail
            history = [{"role": last["role"], "text": last_text, "ts": last["ts"]}]
    _save_history(user_id, history)


@app.post("/process")
async def process_request(request: UserRequest):
    user_id = request.user_id or "anon"
    try:
        # Saving the user's progress
        _append_turn(user_id, "User", request.message)

        # Forming context (latest moves)
        history = _load_history(user_id)
        history_text = "\n".join([f"{h['role']}: {h['text']}" for h in history])

        # Create a single entry point for run_crew
        # Insert an explicit “Dialogue context” label — the agent manager will see the background information.
        combined_input = (
            "Контекст диалога (последние реплики):\n"
            f"{history_text}\n\n"
            "Текущее сообщение пользователя (ниже):\n"
            f"{request.message}"
        )

        # launch crew
        response = await run_crew(combined_input)

        # run_crew returns res.raw. Let's play it safe: if it's a dict, we'll get the string
        if isinstance(response, dict):
            assistant_text = response.get("response") or response.get("result") or json.dumps(response, ensure_ascii=False)
        else:
            assistant_text = str(response)

        # Save the assistant's response in history
        _append_turn(user_id, "Assistant", assistant_text)

        return {"response": assistant_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(settings.RAG_CREWAI_URL.split(":")[-1].split("/")[0])
    )