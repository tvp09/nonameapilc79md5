import os
import json
import math
import hashlib
import asyncio
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# CONFIG
# ============================================================

SOURCE_API = os.getenv(
    "SOURCE_API",
    "https://wtxmd52.tele68.com/v1/txmd5/lite-sessions?cp=R&cl=R&pf=web&at=6256356b3f81d9bd5e0950ecd13c7df0",
)

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "5"))
MAX_HISTORY = 200
ADMIN = os.getenv("ADMIN", "noname")
API_KEY = os.getenv("API_KEY", "").strip()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_FILE = DATA_DIR / "history.json"
STATE_FILE = DATA_DIR / "model_state.json"
PREDICTIONS_FILE = DATA_DIR / "predictions.json"

app = FastAPI(
    title="Sic Bo Prediction API",
    version="1.0.0",
    description="Render API trung gian: lấy dữ liệu, lưu 200 phiên và dự đoán Tài/Xỉu.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

http_client = httpx.AsyncClient(timeout=15.0, follow_redirects=True)

# ============================================================
# JSON STORAGE
# ============================================================

def load_json(path: Path, default):
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def save_json(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_history():
    return load_json(HISTORY_FILE, [])


def save_history(history):
    # oldest -> newest
    history = sorted(history, key=lambda x: int(x["session"]))
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    save_json(HISTORY_FILE, history)


def load_state():
    return load_json(
        STATE_FILE,
        {
            "bandit": {
                "bayes": 1.0,
                "hmm": 1.0,
                "md5": 1.0,
            },
            "accuracy": {
                "total": 0,
                "correct": 0,
                "wrong": 0,
            },
            "last_source_session": None,
            "updated_at": None,
        },
    )


def save_state(state):
    save_json(STATE_FILE, state)


def load_predictions():
    return load_json(PREDICTIONS_FILE, {})


def save_predictions(predictions):
    # Giữ lại lịch sử dự đoán liên quan tới 200 phiên gần nhất + một ít phiên kế tiếp.
    if len(predictions) > 400:
        items = sorted(predictions.items(), key=lambda kv: int(kv[0]))
        predictions = dict(items[-400:])
    save_json(PREDICTIONS_FILE, predictions)


# ============================================================
# ALGORITHM ENGINE
# ============================================================

class BayesianModel:
    def probability(self, seq):
        # Rebuild từ lịch sử để không bị cộng lặp sau restart.
        alpha = 1.0
        beta = 1.0
        for result in seq:
            if result == 1:
                alpha += 1
            else:
                beta += 1
        return alpha / (alpha + beta)


class HMMModel:
    def predict(self, seq):
        if len(seq) < 8:
            return 0

        last = seq[-1]
        score = 0.0

        # Cầu bệt
        streak = 1
        for i in range(len(seq) - 2, -1, -1):
            if seq[i] == last:
                streak += 1
            else:
                break

        if streak >= 4:
            score += min(1.2, streak * 0.25)

        # Cầu đảo đều
        if seq[-6:] == [1, 0] * 3 or seq[-6:] == [0, 1] * 3:
            score -= 1.0

        # Cầu gãy
        recent = seq[-7:-1]
        if recent.count(last) <= 2:
            score -= 0.8

        # Nhiễu cao
        if len(set(seq[-6:])) == 2 and streak <= 2:
            score *= 0.5

        if score > 0.4:
            return 1
        elif score < -0.4:
            return -1
        return 0


class Bandit:
    def __init__(self, state):
        b = state.setdefault("bandit", {})
        self.score = defaultdict(float)
        for name in ("bayes", "hmm", "md5"):
            self.score[name] = float(b.get(name, 1.0))

    def weight(self, name):
        return max(0.3, min(1.5, self.score[name]))

    def reward(self, name, win):
        self.score[name] *= 0.95
        if win:
            self.score[name] += 0.15
        else:
            self.score[name] -= 0.20

    def export(self):
        return {
            k: round(max(0.3, min(1.5, float(v))), 6)
            for k, v in self.score.items()
        }


def entropy(seq, w=20):
    if len(seq) < w:
        return 0.0

    win = seq[-w:]
    p1 = sum(win) / w
    p0 = 1.0 - p1

    if p1 in (0, 1):
        return 0.0

    return -(p1 * math.log2(p1) + p0 * math.log2(p0))


def chaos_block(seq):
    if len(seq) < 25:
        return False
    return entropy(seq, 30) > 0.92


def algo_md5(_id, seq_len):
    md5 = hashlib.md5(str(_id).encode()).hexdigest()
    sig = 1 if int(md5[-1], 16) < 8 else -1

    if seq_len > 40:
        sig *= 0.4

    return sig


def sigmoid(x):
    # tránh overflow khi score cực lớn
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def calculate_final(seq, session_id, state):
    bayes = BayesianModel()
    hmm = HMMModel()
    bandit = Bandit(state)

    n = len(seq)
    caution = chaos_block(seq)

    p_bayes = bayes.probability(seq)
    hmm_sig = hmm.predict(seq)
    md5_sig = algo_md5(session_id, n)

    w_bayes = bandit.weight("bayes")
    w_hmm = bandit.weight("hmm")
    w_md5 = bandit.weight("md5")

    if n < 25:
        score = (
            (p_bayes - 0.5) * 4
            + hmm_sig * 3 * w_hmm
            + md5_sig * 2 * w_md5
        )
    else:
        score = (
            (p_bayes - 0.5) * 6 * w_bayes
            + hmm_sig * 2 * w_hmm
            + md5_sig * 1 * w_md5
        )

    # Khi chaos cao, giảm tác động để confidence không bị thổi phồng.
    if caution:
        score *= 0.75

    prob_tai = sigmoid(score)
    final = "TÀI" if prob_tai >= 0.5 else "XỈU"

    confidence = round(max(prob_tai, 1.0 - prob_tai) * 100.0, 2)
    return {
        "final": final,
        "confidence": confidence,
        "tai_probability": round(prob_tai * 100.0, 2),
        "xiu_probability": round((1.0 - prob_tai) * 100.0, 2),
        "score": round(score, 6),
        "caution": caution,
        "bayesian_probability": round(p_bayes * 100.0, 2),
        "hmm_signal": hmm_sig,
        "md5_signal": md5_sig,
        "bandit": bandit.export(),
        "entropy": round(entropy(seq, min(30, max(20, len(seq)))) if len(seq) >= 20 else 0.0, 6),
    }


# ============================================================
# SOURCE API / NORMALIZATION
# ============================================================

def normalize_result(raw):
    try:
        session = int(raw.get("id"))
        dice = raw.get("dices") or []
        point = int(raw.get("point"))
        result_raw = str(raw.get("resultTruyenThong", "")).upper().strip()

        if len(dice) != 3:
            return None

        dice = [int(x) for x in dice]
        if any(x < 1 or x > 6 for x in dice):
            return None

        # API gốc trả XIU / TAI. Nếu thiếu result thì tính theo point.
        if result_raw in ("TAI", "TÀI"):
            result = "TÀI"
        elif result_raw in ("XIU", "XỈU"):
            result = "XỈU"
        else:
            result = "TÀI" if point >= 11 else "XỈU"

        return {
            "session": session,
            "dice": dice,
            "point": point,
            "result": result,
        }
    except Exception:
        return None


async def fetch_source():
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 SicBoRenderAPI/1.0",
    }

    response = await http_client.get(SOURCE_API, headers=headers)
    response.raise_for_status()
    payload = response.json()

    items = payload.get("list", [])
    if not isinstance(items, list):
        return []

    normalized = []
    for item in items:
        row = normalize_result(item)
        if row:
            normalized.append(row)

    # oldest -> newest
    normalized.sort(key=lambda x: x["session"])
    return normalized


# ============================================================
# HISTORY + PREDICTION TRACKING
# ============================================================

def result_to_bit(result):
    return 1 if result == "TÀI" else 0


def update_history_from_source(source_rows):
    history = load_history()
    existing = {int(x["session"]): x for x in history}

    for row in source_rows:
        existing[int(row["session"])] = row

    merged = list(existing.values())
    merged.sort(key=lambda x: int(x["session"]))

    if len(merged) > MAX_HISTORY:
        merged = merged[-MAX_HISTORY:]

    save_history(merged)
    return merged


def evaluate_predictions(history, state):
    """
    Mỗi prediction key là session mà mô hình đã dự đoán.
    Khi session đó xuất hiện trong history, đối chiếu đúng/sai
    và reward Bandit đúng 1 lần.
    """
    predictions = load_predictions()
    history_map = {int(x["session"]): x for x in history}

    total = int(state["accuracy"].get("total", 0))
    correct = int(state["accuracy"].get("correct", 0))
    wrong = int(state["accuracy"].get("wrong", 0))

    changed = False
    bandit = Bandit(state)

    for session_key, pred in list(predictions.items()):
        try:
            session_id = int(session_key)
        except Exception:
            continue

        if pred.get("evaluated"):
            continue

        actual = history_map.get(session_id)
        if not actual:
            continue

        predicted_result = pred.get("prediction")
        actual_result = actual.get("result")

        if predicted_result not in ("TÀI", "XỈU"):
            pred["evaluated"] = True
            continue

        win = predicted_result == actual_result

        total += 1
        if win:
            correct += 1
        else:
            wrong += 1

        # Reward từng thành phần dựa trên tín hiệu mà prediction đã lưu.
        component_signals = pred.get("component_signals", {})
        actual_bit = result_to_bit(actual_result)

        for name in ("bayes", "hmm", "md5"):
            sig = component_signals.get(name)
            if sig is None:
                continue

            if name == "bayes":
                component_pred = "TÀI" if float(sig) >= 50 else "XỈU"
            else:
                component_pred = "TÀI" if float(sig) > 0 else "XỈU"

            # HMM 0 là trung tính: không reward/penalty.
            if name == "hmm" and int(sig) == 0:
                continue

            bandit.reward(name, component_pred == actual_result)

        pred["evaluated"] = True
        pred["actual"] = actual_result
        pred["correct"] = win
        changed = True

    state["bandit"] = bandit.export()
    state["accuracy"] = {
        "total": total,
        "correct": correct,
        "wrong": wrong,
    }

    if changed:
        save_predictions(predictions)

    return state


def create_prediction(history):
    if not history:
        return None

    seq = [result_to_bit(x["result"]) for x in history]
    current = history[-1]
    next_session = int(current["session"]) + 1

    state = load_state()
    result = calculate_final(seq, next_session, state)

    predictions = load_predictions()
    predictions[str(next_session)] = {
        "session": next_session,
        "prediction": result["final"],
        "confidence": result["confidence"],
        "component_signals": {
            "bayes": result["bayesian_probability"],
            "hmm": result["hmm_signal"],
            "md5": result["md5_signal"],
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluated": False,
    }

    save_predictions(predictions)

    return {
        "session": next_session,
        "prediction": result["final"],
        "confidence": result["confidence"],
        "tai_probability": result["tai_probability"],
        "xiu_probability": result["xiu_probability"],
        "engine": result,
    }


def build_prediction_for_current_history():
    history = load_history()
    if not history:
        return None

    # Không cần tạo trùng nếu đã có dự đoán cho session kế tiếp.
    next_session = int(history[-1]["session"]) + 1
    predictions = load_predictions()

    existing = predictions.get(str(next_session))
    if existing and not existing.get("evaluated", False):
        return existing

    return create_prediction(history)


# ============================================================
# POLLER
# ============================================================

last_processed_new_session = None


async def sync_once():
    global last_processed_new_session

    source_rows = await fetch_source()
    if not source_rows:
        return {"new": False, "reason": "empty_source"}

    old_history = load_history()
    old_ids = {int(x["session"]) for x in old_history}

    history = update_history_from_source(source_rows)
    new_rows = [x for x in history if int(x["session"]) not in old_ids]

    state = load_state()
    state = evaluate_predictions(history, state)

    latest_session = int(history[-1]["session"])
    state["last_source_session"] = latest_session
    state["updated_at"] = datetime.now(timezone.utc).isoformat()

    # Chỉ tạo prediction mới khi có phiên mới hoặc chưa có prediction.
    if new_rows or not load_predictions().get(str(latest_session + 1)):
        create_prediction(history)

    save_state(state)

    if new_rows:
        last_processed_new_session = latest_session

    return {
        "new": bool(new_rows),
        "new_sessions": [x["session"] for x in new_rows],
        "latest": latest_session,
        "history_count": len(history),
    }


async def poll_loop():
    while True:
        try:
            await sync_once()
        except Exception as e:
            # Không spam log bình thường; chỉ log lỗi.
            print(f"[ERROR] sync: {type(e).__name__}: {e}")
        await asyncio.sleep(POLL_SECONDS)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(poll_loop())


@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()


# ============================================================
# AUTH
# ============================================================

def check_api_key(x_api_key: str | None):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {
        "success": True,
        "name": "Sic Bo Prediction API",
        "admin": ADMIN,
        "endpoints": [
            "/api/status",
            "/api/latest",
            "/api/predict",
            "/api/history",
            "/api/stats",
        ],
    }


@app.get("/api/status")
async def status(x_api_key: str | None = Header(default=None)):
    check_api_key(x_api_key)
    history = load_history()
    state = load_state()

    return {
        "success": True,
        "status": "online",
        "source": "connected",
        "phienhientai": history[-1]["session"] if history else None,
        "history": len(history),
        "max_history": MAX_HISTORY,
        "poll_seconds": POLL_SECONDS,
        "accuracy": state["accuracy"],
        "admin": ADMIN,
    }


@app.get("/api/latest")
async def latest(x_api_key: str | None = Header(default=None)):
    check_api_key(x_api_key)
    history = load_history()

    if not history:
        await sync_once()
        history = load_history()

    if not history:
        raise HTTPException(status_code=503, detail="No history data")

    return {
        "success": True,
        "data": history[-1],
        "admin": ADMIN,
    }


@app.get("/api/predict")
async def predict(x_api_key: str | None = Header(default=None)):
    check_api_key(x_api_key)

    history = load_history()
    if not history:
        await sync_once()
        history = load_history()

    if not history:
        raise HTTPException(status_code=503, detail="No source data")

    # Đánh giá prediction cũ trước khi trả prediction mới.
    state = evaluate_predictions(history, load_state())
    save_state(state)

    prediction = build_prediction_for_current_history()
    if not prediction:
        raise HTTPException(status_code=503, detail="Prediction unavailable")

    current = history[-1]

    return {
        "success": True,
        "data": {
            "phienhientai": current["session"],
            "xucxac": {
                "dice": current["dice"],
                "point": current["point"],
                "result": current["result"],
            },
            "dudoanphien": prediction["session"],
            "dudoan": prediction["prediction"],
            "dotincay": f'{float(prediction["confidence"]):.2f}%',
            "tai_probability": prediction.get("tai_probability"),
            "xiu_probability": prediction.get("xiu_probability"),
            "algorithm": prediction.get("engine", {}),
            "storage": {
                "current": len(history),
                "max": MAX_HISTORY,
            },
            "admin": ADMIN,
        },
    }


@app.get("/api/history")
async def history_api(
    limit: int = 200,
    x_api_key: str | None = Header(default=None),
):
    check_api_key(x_api_key)

    limit = max(1, min(MAX_HISTORY, limit))
    history = load_history()

    return {
        "success": True,
        "count": min(len(history), limit),
        "max": MAX_HISTORY,
        "data": history[-limit:],
        "admin": ADMIN,
    }


@app.get("/api/stats")
async def stats(x_api_key: str | None = Header(default=None)):
    check_api_key(x_api_key)

    history = load_history()
    state = load_state()

    tai = sum(1 for x in history if x["result"] == "TÀI")
    xiu = sum(1 for x in history if x["result"] == "XỈU")
    total = len(history)

    accuracy_total = int(state["accuracy"].get("total", 0))
    accuracy_correct = int(state["accuracy"].get("correct", 0))
    accuracy = (
        round(accuracy_correct / accuracy_total * 100, 2)
        if accuracy_total
        else 0.0
    )

    return {
        "success": True,
        "data": {
            "history_count": total,
            "tai": tai,
            "xiu": xiu,
            "tyletai": round(tai / total * 100, 2) if total else 0,
            "tylexiu": round(xiu / total * 100, 2) if total else 0,
            "prediction_tracking": {
                "total": accuracy_total,
                "correct": accuracy_correct,
                "wrong": int(state["accuracy"].get("wrong", 0)),
                "accuracy": accuracy,
            },
            "bandit": state.get("bandit", {}),
        },
        "admin": ADMIN,
    }
