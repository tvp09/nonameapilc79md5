import os
import json
import math
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
    title="Tai Xiu Analysis API",
    version="1.0.0",
    description="Render API trung gian: lấy dữ liệu, lưu 200 phiên và phân tích Tài/Xỉu.",
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
                "markov": 1.0,
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
# ALGORITHM ENGINE - TÀI/XỈU ENSEMBLE
# ============================================================
#
# 4 thành phần hiện có được GIỮ và nâng cấp:
#   1) BayesianModel  -> Bayesian thích nghi theo độ mới.
#   2) HMMModel       -> HMM/pattern state, chuyển trạng thái + streak.
#   3) Bandit         -> tự điều chỉnh trọng số theo kết quả thực tế.
#   4) entropy/chaos  -> phát hiện regime/nhiễu để giảm confidence.
#
# MD5 prediction đã được LOẠI BỎ vì mã phiên không phải tín hiệu thống kê
# đáng tin cậy cho kết quả Tài/Xỉu.
#
# Thuật toán mới thay MD5:
#   MarkovModel -> học xác suất chuyển Tài/Xỉu bậc 1 và bậc 2.
#
# Tất cả mô hình chỉ dùng tối đa 200 phiên đang lưu, không tạo dữ liệu giả.
# ============================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def weighted_rate(seq, decay=0.965):
    """Tỷ lệ Tài có trọng số, phiên mới quan trọng hơn phiên cũ."""
    if not seq:
        return 0.5

    weight = 1.0
    tai_weight = 0.0
    total_weight = 0.0

    for value in reversed(seq):
        total_weight += weight
        tai_weight += value * weight
        weight *= decay

    return tai_weight / total_weight if total_weight else 0.5


class BayesianModel:
    """Bayesian thích nghi: prior + tỷ lệ toàn cục + tỷ lệ gần đây."""

    def probability(self, seq):
        if not seq:
            return 0.5

        n = len(seq)

        # Prior yếu, tránh 200 phiên Tài/Xỉu làm mô hình quá cực đoan.
        alpha = 2.0
        beta = 2.0

        # Toàn bộ cửa sổ nhưng giảm dần theo độ cũ.
        decay = 0.975
        weight = 1.0
        for value in reversed(seq):
            if value == 1:
                alpha += weight
            else:
                beta += weight
            weight *= decay

        p_global = alpha / (alpha + beta)

        # Cửa sổ ngắn để bắt thay đổi regime.
        short_n = min(24, n)
        short = seq[-short_n:]
        p_short = (2.0 + sum(short)) / (4.0 + short_n)

        # Cửa sổ trung bình ổn định hơn cửa sổ quá ngắn.
        mid_n = min(80, n)
        mid = seq[-mid_n:]
        p_mid = (4.0 + sum(mid)) / (8.0 + mid_n)

        # Nếu dữ liệu ngắn thì ưu tiên toàn cục; dữ liệu đủ 200 thì
        # tăng trọng số tín hiệu gần đây nhưng vẫn shrink về 50%.
        if n < 12:
            p = 0.65 * p_global + 0.35 * p_short
        elif n < 40:
            p = 0.45 * p_global + 0.35 * p_mid + 0.20 * p_short
        else:
            p = 0.30 * p_global + 0.30 * p_mid + 0.40 * p_short

        # Shrink nhẹ về 50% để chống overfit trên chuỗi ngắn/nhiễu.
        shrink = 0.08 if n >= 40 else 0.15
        return clamp(0.5 + (p - 0.5) * (1.0 - shrink), 0.08, 0.92)


class HMMModel:
    """
    HMM/pattern-state cải tiến.
    Không giả định bệt luôn tiếp diễn: đánh giá đồng thời
    streak, chuyển trạng thái và các mẫu gần đây.
    """

    def predict(self, seq):
        if len(seq) < 8:
            return 0

        n = len(seq)
        last = seq[-1]

        # --- State 1: streak ---
        streak = 1
        for i in range(n - 2, -1, -1):
            if seq[i] == last:
                streak += 1
            else:
                break

        # Xác suất tiếp tục bệt từ các đoạn trong lịch sử có cùng độ dài.
        continuation = 0.5
        if streak >= 2:
            hits = total = 0
            for i in range(streak, n):
                if seq[i - streak:i] == [last] * streak:
                    total += 1
                    hits += seq[i] == last
            if total:
                continuation = (hits + 2.0) / (total + 4.0)

        # --- State 2: nhịp đảo ---
        alt_len = min(10, n)
        alt = seq[-alt_len:]
        alternations = sum(alt[i] != alt[i - 1] for i in range(1, len(alt)))
        alt_ratio = alternations / max(1, len(alt) - 1)

        # --- State 3: mẫu cuối 2-4 phiên ---
        pattern_score = 0.0
        for k, strength in ((4, 0.45), (3, 0.35), (2, 0.20)):
            if n <= k:
                continue
            pat = tuple(seq[-k:])
            matches = []
            for i in range(k, n):
                if tuple(seq[i-k:i]) == pat:
                    matches.append(seq[i])
            if matches:
                p = (sum(matches) + 2.0) / (len(matches) + 4.0)
                pattern_score += (p - 0.5) * strength

        # Chấm điểm về phía Tài.
        score = 0.0

        # Streak chỉ được dùng khi lịch sử thực sự ủng hộ continuation.
        if streak >= 3:
            score += clamp((continuation - 0.5) * 4.0, -0.9, 0.9)

        # Chuỗi đảo mạnh thì ưu tiên đảo theo nhịp, nhưng không tuyệt đối.
        if alt_ratio >= 0.80:
            score += (-0.55 if last == 1 else 0.55)

        score += pattern_score

        # Nếu đoạn gần nhất cực kỳ cân bằng, giảm tín hiệu HMM.
        recent = seq[-8:]
        balance = abs(sum(recent) / len(recent) - 0.5)
        if balance < 0.125:
            score *= 0.65

        if score > 0.22:
            return 1
        if score < -0.22:
            return -1
        return 0


class MarkovModel:
    """Thuật toán mới thay MD5: Markov bậc 1 + bậc 2."""

    def probability(self, seq):
        if len(seq) < 4:
            return 0.5

        # Bậc 1: P(next | last)
        counts1 = {
            0: [1.5, 1.5],  # [xiu, tai]
            1: [1.5, 1.5],
        }

        # Bậc 2: P(next | last2)
        counts2 = defaultdict(lambda: [1.25, 1.25])

        # Ưu tiên các transition gần đây.
        for i in range(1, len(seq)):
            context = seq[i - 1]
            nxt = seq[i]
            age = len(seq) - 1 - i
            weight = 0.985 ** age
            counts1[context][nxt] += weight

            if i >= 2:
                ctx2 = (seq[i - 2], seq[i - 1])
                counts2[ctx2][nxt] += weight

        last = seq[-1]
        c1 = counts1[last]
        p1 = c1[1] / sum(c1)

        ctx2 = tuple(seq[-2:])
        c2 = counts2.get(ctx2)
        p2 = c2[1] / sum(c2) if c2 else p1

        # Mẫu bậc 2 được ưu tiên khi đủ quan sát.
        p = 0.42 * p1 + 0.58 * p2

        # Shrink để tránh một pattern hiếm làm xác suất nhảy quá mạnh.
        return clamp(0.5 + (p - 0.5) * 0.78, 0.12, 0.88)


class Bandit:
    """Adaptive ensemble: trọng số dựa trên hiệu quả thực tế."""

    COMPONENTS = ("bayes", "hmm", "markov")

    def __init__(self, state):
        b = state.setdefault("bandit", {})
        self.score = defaultdict(lambda: 1.0)

        for name in self.COMPONENTS:
            value = float(b.get(name, 1.0))
            self.score[name] = clamp(value, 0.35, 1.8)

    def weight(self, name):
        return clamp(self.score[name], 0.35, 1.8)

    def reward(self, name, win):
        # Learning rate vừa phải để 1 phiên sai không làm mô hình mất niềm tin.
        old = self.score[name]
        if win:
            new = old + 0.10 * (1.8 - old)
        else:
            new = old - 0.075 * (old - 0.35)

        # EMA nhẹ chống dao động.
        self.score[name] = clamp(0.92 * old + 0.08 * new, 0.35, 1.8)

    def export(self):
        return {
            k: round(clamp(float(v), 0.35, 1.8), 6)
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


def regime_strength(seq):
    """
    Đo mức ổn định của cửa sổ gần đây.
    Gần 0 = cân bằng/nhiễu; gần 1 = lệch mạnh.
    Đây là bộ lọc confidence, không tự quyết định kết quả.
    """
    if len(seq) < 20:
        return 0.0

    recent = seq[-20:]
    balance = abs(sum(recent) / 20.0 - 0.5) * 2.0

    transitions = sum(
        recent[i] != recent[i - 1]
        for i in range(1, len(recent))
    ) / 19.0

    # Transition khoảng 0.5 là trạng thái khó đoán.
    transition_stability = abs(transitions - 0.5) * 2.0

    return clamp(0.70 * balance + 0.30 * transition_stability, 0.0, 1.0)


def chaos_block(seq):
    if len(seq) < 25:
        return False

    h = entropy(seq, min(30, len(seq)))
    strength = regime_strength(seq)

    # Entropy cao + regime strength thấp => giảm confidence.
    return h > 0.96 and strength < 0.25


def sigmoid(x):
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def calculate_final(seq, session_id, state):
    """Ensemble 4 lớp, có hiệu chỉnh lệch Tài/Xỉu.

    Giữ nguyên Bayesian + HMM + Bandit + bộ lọc entropy/regime,
    đồng thời dùng Markov làm tín hiệu chuyển trạng thái thay cho MD5.
    Điểm quan trọng: không cho một tín hiệu Tài đơn lẻ kéo cả ensemble
    sang Tài khi tỷ lệ thực tế gần đây đang nghiêng về Xỉu.
    """
    del session_id

    bayes = BayesianModel()
    hmm = HMMModel()
    markov = MarkovModel()
    bandit = Bandit(state)

    n = len(seq)
    caution = chaos_block(seq)

    p_bayes = bayes.probability(seq)
    hmm_sig = hmm.predict(seq)
    p_markov = markov.probability(seq)

    w_bayes = bandit.weight("bayes")
    w_hmm = bandit.weight("hmm")
    w_markov = bandit.weight("markov")

    # Tín hiệu liên tục của Bayesian + Markov.
    bayes_score = (p_bayes - 0.5) * 2.0
    markov_score = (p_markov - 0.5) * 2.0
    hmm_score = float(hmm_sig)

    total_w = w_bayes + w_markov + w_hmm
    if total_w <= 0:
        total_w = 1.0

    ensemble = (
        bayes_score * w_bayes
        + markov_score * w_markov
        + hmm_score * w_hmm
    ) / total_w

    # --------------------------------------------------------
    # 1) BASE-RATE CALIBRATION
    # --------------------------------------------------------
    # Neo mô hình theo tỷ lệ thật của 30/80 phiên gần nhất.
    # Đây là phần chống lỗi "toàn Tài" khi dữ liệu thực tế đang nghiêng Xỉu.
    recent_n = min(30, n)
    mid_n = min(80, n)
    recent_rate = sum(seq[-recent_n:]) / recent_n if recent_n else 0.5
    mid_rate = sum(seq[-mid_n:]) / mid_n if mid_n else 0.5
    global_rate = sum(seq) / n if n else 0.5

    # Cửa sổ gần đây được ưu tiên nhưng không bỏ qua 200 phiên.
    base_rate = 0.55 * recent_rate + 0.30 * mid_rate + 0.15 * global_rate
    base_score = (base_rate - 0.5) * 2.0

    # Neo nhẹ về base-rate. Khi base-rate Xỉu, tín hiệu Tài phải có
    # bằng chứng đủ mạnh mới vượt 50%.
    ensemble = 0.72 * ensemble + 0.28 * base_score

    # --------------------------------------------------------
    # 2) MOMENTUM NHỎ, KHÔNG ĐƯỢC ÁP ĐẢO
    # --------------------------------------------------------
    momentum_n = min(12, n)
    momentum_rate = sum(seq[-momentum_n:]) / momentum_n if momentum_n else 0.5
    momentum = (momentum_rate - 0.5) * 0.18

    # --------------------------------------------------------
    # 3) ANTI-BIAS GUARD
    # --------------------------------------------------------
    # Nếu ensemble nghiêng Tài nhưng dữ liệu gần đây nghiêng Xỉu,
    # giảm mạnh độ lệch. Ngược lại cũng áp dụng đối xứng cho Xỉu.
    disagreement = abs(ensemble - base_score)
    if disagreement > 0.32:
        ensemble = 0.62 * ensemble + 0.38 * base_score

    score = ensemble * 2.0 + momentum

    # Dữ liệu ít -> bảo thủ.
    if n < 12:
        score *= 0.45
    elif n < 25:
        score *= 0.72

    # Khi chuỗi có entropy cao, giảm confidence.
    if caution:
        score *= 0.58

    prob_tai = sigmoid(score)

    # --------------------------------------------------------
    # 4) SYMMETRIC CONFIDENCE CAP
    # --------------------------------------------------------
    # Không cho bên nào (Tài/Xỉu) được ưu ái. Cap đối xứng.
    max_conf = 0.82 if caution else 0.86
    prob_tai = 0.5 + (prob_tai - 0.5) * (max_conf - 0.5) / 0.5
    prob_tai = clamp(prob_tai, 1.0 - max_conf, max_conf)

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
        "markov_probability": round(p_markov * 100.0, 2),
        "hmm_signal": hmm_sig,
        "bandit": bandit.export(),
        "recent_tai_rate": round(recent_rate * 100.0, 2),
        "mid_tai_rate": round(mid_rate * 100.0, 2),
        "global_tai_rate": round(global_rate * 100.0, 2),
        "calibration_base_tai": round(base_rate * 100.0, 2),
        "entropy": round(
            entropy(seq, min(30, max(20, len(seq))))
            if len(seq) >= 20 else 0.0,
            6,
        ),
        "regime_strength": round(regime_strength(seq), 6),
        "history_used": min(n, MAX_HISTORY),
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
        "User-Agent": "Mozilla/5.0 TaiXiuRenderAPI/2.0",
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

        for name in ("bayes", "hmm", "markov"):
            sig = component_signals.get(name)
            if sig is None:
                continue

            if name == "bayes" or name == "markov":
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
            "markov": result["markov_probability"],
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
        "name": "Tai Xiu Analysis API",
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
