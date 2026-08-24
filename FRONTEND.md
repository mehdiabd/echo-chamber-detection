# تاریخچه ران‌ها: زمان باقی‌مانده، توقف، ری‌استارت

پایهٔ لوکال: `http://localhost:8765`  
پایهٔ پروداکشن: `https://api.synappse.ir/api/echo-chamber`

برای بخش تاریخچه ران‌ها در صفحهٔ `/pipeline`.

هر ۲ تا ۳ ثانیه `GET /api/v1/pipeline/runs` یا `GET /api/v1/pipeline/runs/{run_id}` را poll کن.

```json
{
  "run_id": "…",
  "status": "running",
  "progress": {
    "percent": 22,
    "remaining_percent": 78,
    "remaining_seconds": 780,
    "remaining_label": "حدود 13 دقیقه",
    "elapsed_seconds": 240,
    "docs_done": 400,
    "docs_total": 8000,
    "stage": "fetch",
    "message": "اسکن اولیه الستیک — 400/8000 سند"
  },
  "actions": {
    "can_stop": true,
    "can_restart": true
  },
  "error": null
}
```

| فیلد | UI |
|---|---|
| `progress.remaining_label` | متن زمان مانده؛ کنار وضعیت ران نشان بده |
| `progress.remaining_seconds` | ثانیهٔ مانده؛ اگر هنوز قابل تخمین نباشد `null` |
| `progress.elapsed_seconds` | زمان گذشته از شروع |
| `progress.docs_done` / `docs_total` | پیشرفت اسکن الستیک (اگر در مرحلهٔ دریافت باشد) |
| `status` | `queued` `running` `done` `failed` `cancelled` |
| `actions.can_stop` | دکمهٔ توقف |
| `actions.can_restart` | دکمهٔ اجرای دوباره |

اگر `remaining_seconds === null` همان `در حال محاسبه` را نشان بده.

## توقف

`POST /api/v1/pipeline/runs/{run_id}/stop`

فقط برای `queued` یا `running`. پاسخ `200` با `status: "cancelled"`. اگر ران تمام شده باشد `409`.

## ری‌استارت

`POST /api/v1/pipeline/runs/{run_id}/restart`

اگر ران هنوز فعال باشد اول متوقف می‌شود، بعد یک ران جدید با همان پارامترها ساخته می‌شود. پاسخ `202` به‌علاوهٔ `restarted_from`. اگر ران دیگری در حال اجرا باشد `409`.

```json
{
  "run_id": "…",
  "status": "queued",
  "restarted_from": "2568abb5131440e594e09d16283f888d",
  "progress": {
    "percent": 0,
    "remaining_seconds": null,
    "remaining_label": "در حال محاسبه",
    "stage": "queued",
    "message": "در صف"
  },
  "actions": {"can_stop": true, "can_restart": true}
}
```
