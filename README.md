# Sic Bo Prediction API - Render

## Chạy local

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Mở:
- `/docs`
- `/api/status`
- `/api/predict`
- `/api/history`
- `/api/stats`

## Render

1. Đưa `app.py`, `requirements.txt`, `render.yaml` lên GitHub.
2. Tạo Web Service trên Render.
3. Build command:
   `pip install -r requirements.txt`
4. Start command:
   `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Có thể đặt `API_KEY` trong Environment Variables.
6. `ADMIN` mặc định là `noname`.

## Lưu ý storage

Code lưu JSON trong thư mục `data/` và giữ tối đa 200 phiên.
Render Web Service thông thường có filesystem tạm thời; nếu cần dữ liệu sống lâu qua restart/deploy,
nên chuyển storage sang PostgreSQL/Redis hoặc persistent disk phù hợp với gói Render.

## API

GET `/api/predict`

Ví dụ:

```json
{
  "success": true,
  "data": {
    "phienhientai": 7078571,
    "xucxac": {
      "dice": [1, 4, 4],
      "point": 9,
      "result": "XỈU"
    },
    "dudoanphien": 7078572,
    "dudoan": "TÀI",
    "dotincay": "67.42%",
    "admin": "noname"
  }
}
```

Nếu đặt API_KEY, gửi header:
`X-API-Key: YOUR_KEY`
