# راهنمای فرانت‌اند — Echo Chamber

این فایل نقشهٔ کار است: برای هر صفحه ببین کجا را بخوانی، کدام API را صدا بزنی، و کدام فایل نمونه را نگاه کنی.

پایهٔ لوکال: `http://localhost:8765`  
قرارداد زنده (Swagger): [http://localhost:8765/docs](http://localhost:8765/docs)  
OpenAPI برای Postman: [http://localhost:8765/openapi.json](http://localhost:8765/openapi.json) → Import → Link

---

## راه‌اندازی با Docker

`make docker-up` **فقط API را بالا می‌آورد.** Elasticsearch داخل این استک نیست و داشبوردها هم در git نیستند (`*.html` / `*.json`). برای سیم‌کشی فرانت همین کافی است: `/docs` و `/health` باید جواب بدهند حتی اگر لیست داشبورد خالی باشد.

```bash
cp .env.example .env          # اگر .env نداری
make docker-up                # API روی :8765
open http://localhost:8765/docs
```

بدون Makefile، `docker compose up` دیگر به‌خاطر نبود `.env` نمی‌ترکد؛ باز هم `make docker-up` را ترجیح بده چون `.env` را می‌سازد.

### داده از کجا می‌آید؟

سه راه، به ترتیب ساده‌به‌سخت:

1. **کپی آرتیفکت از ماشینی که پایپلاین را ران کرده** (بهترین برای فرانت): `timeline_dashboard.html`، چند `dashboard_*.html` + `hybrid_graph_*.html` + `*_legend.json`، `communities/hybrid_report_*.json`، و در صورت نیاز `lib/bindings/utils.js`. بعد دوباره `make docker-up`. API همان فایل‌ها را از ریشهٔ پروژه می‌خواند.
2. **واکشی از Elasticsearch و ساخت داشبورد** — فقط روی شبکه/VPN سازمان، با فایل گواهی:
   ```bash
   # ELASTIC_AUTH=1 → ca.crt و https://192.168.59.79:9200
   # ELASTIC_AUTH=2 → http_ca.crt و https://192.168.59.26:9200
   # در .env: TOPIC_LABEL, START_DATE, END_DATE یا LOOKBACK_DAYS
   make docker-elastic     # می‌نویسد interactions.json و res.json و pipeline_config.json
   make docker-pipeline    # از آن فایل‌ها HTML/گزارش می‌سازد
   ```
   یا از خود API: `POST /api/v1/pipeline/runs` با `"fetch": true, "detect": true` — همان `elastic.py` را داخل کانتینر `api` اجرا می‌کند و همان گواهی/VPN را می‌خواهد.
3. **بدون داده کار کردن:** `/docs` و قرارداد API برای پیاده‌سازی UI کافی است؛ `GET /api/v1/dashboards` تا وقتی آرتیفکت نباشد `[]` برمی‌گردد.

`make docker-elastic` اگر `ca.crt` / `http_ca.crt` نباشد عمداً خطا می‌دهد. این فایل‌ها gitignore شده‌اند؛ باید جداگانه کنار پروژه گذاشته شوند.

جزئیات CLIی `elastic.py`: موضوع، بازهٔ تاریخ، و خروجی‌ها در [README.md](README.md#elasticsearch-data-fetch).

---

## از کجا شروع کنی

| چیزی که می‌خواهی | کجاست |
|---|---|
| لیست و شکل همهٔ APIها | `/docs` یا `/openapi.json` |
| نمونهٔ UI تایم‌لاین (استاتیک) | `timeline_dashboard.html` + `timeline_template.html` |
| گراف یک اسلات | `dashboard_*.html` → iframe به `hybrid_graph_*.html` |
| اجرای API | `make docker-up` یا `make api` |
| گرفتن داده از Elastic | `make docker-elastic` سپس `make docker-pipeline` (VPN + `ca.crt`) |
| توکن اختیاری | `.env` → `ECHO_API_TOKEN` یا `PARTY_CHANGE_API_TOKEN` |
| تست قرارداد API | `tests/test_echo_chamber_api.py` |

`timeline_dashboard.html` دست‌نویس نیست و API صدا نمی‌زند. داده داخل HTML embed شده. برای اپ جدید از API استفاده کن، نه scrape.

---

## قرارداد مشترک

- JSON، UTF-8، CORS باز (`Access-Control-Allow-Origin: *`)
- خطا: `{"error": "پیام"}`
- کدها: `400` اعتبارسنجی، `401` بدون توکن، `403` فایل غیرمجاز، `404` نبود، `409` پایپلاین در حال اجرا، `201` ساخت نمونه، `202` شروع پایپلاین
- صفحه‌بندی: `?limit=100&offset=0` (حداکثر `500`)
- اگر توکن ست باشد: هدر `Authorization: Bearer <token>` روی همهٔ `/api/v1/...`
- بدون توکن، `/health` و `/docs` و `/redoc` و `/openapi.json` همیشه بازند

---

## صفحه به صفحه

### ۱) تایم‌لاین / داشبورد (صفحهٔ اصلی)

مرجع UI: `timeline_template.html` (سورس) و `timeline_dashboard.html` (خروجی با دادهٔ واقعی).

| کار UI | API |
|---|---|
| دراپ‌داون موضوع | `GET /api/v1/topics` |
| لیست اسلات‌ها (ساعتی/روزانه/هفتگی/…) | `GET /api/v1/dashboards?slot_mode=daily&start=&end=` |
| جزئیات یک اسلات + لجند + لینک فایل‌ها | `GET /api/v1/dashboards/{slot_id}` |
| HTML گراف / لجند JSON | `GET /api/v1/files/{filename}` |

`slot_id` مثل `daily_260101_to_260102`.

نمونهٔ جزئیات اسلات:

```json
{
  "id": "daily_260101_to_260102",
  "slot_mode": "daily",
  "start": "2026-01-01",
  "end": "2026-01-02",
  "dashboard": "dashboard_daily_260101_to_260102.html",
  "hybrid_graph": "hybrid_graph_daily_260101_to_260102.html",
  "legends": { "hybrid": { "groups": { "alice": 0, "bob": 0 } } },
  "files": {
    "dashboard": "/api/v1/files/dashboard_daily_260101_to_260102.html",
    "legend": "/api/v1/files/dashboard_daily_260101_to_260102_legend.json",
    "hybrid_graph": "/api/v1/files/hybrid_graph_daily_260101_to_260102.html",
    "graph": "/api/v1/files/hybrid_graph_daily_260101_to_260102.json"
  },
  "graph": {
    "node_count": 3,
    "edge_count": 2,
    "nodes": [
      {"id": "alice", "label": "alice", "community": 0, "community_label": "اصولگرا", "color": "#2563eb", "degree": 2, "size": 19}
    ],
    "edges": [
      {"id": "alice|bob", "from": "alice", "to": "bob", "weight": 2}
    ]
  }
}
```

برای رندر گراف در React از `graph.nodes` و `graph.edges` استفاده کن (`from` / `to` سازگار با vis-network). فایل HTML دیگر لازم نیست. این JSON را پایپلاین کنار HTML گراف می‌نویسد؛ آرتیفکت‌های قدیمی که `hybrid_graph_*.json` ندارند `graph.nodes` خالی می‌دهند و باید پایپلاین دوباره اجرا شود (HTML را scrape نکن).

گراف HTML هنوز در `files.hybrid_graph` هست اگر iframe بخواهی. `hybrid_graph_*.html` از CDN (`vis-network` و Bootstrap) و `lib/bindings/utils.js` استفاده می‌کند؛ اگر همان گراف را نشان می‌دهی، از ریشه سرو کن.

**نیست در API:** سایدبار `partyFocus` فقط داخل HTML بaked است. اگر همان سایدبار را می‌خواهی، از HTML نمونه بخوان یا بعداً فیلد جدا به API اضافه می‌شود.

### ۲) گزارش‌ها و جامعهٔ اکو

| کار UI | API |
|---|---|
| لیست گزارش‌ها | `GET /api/v1/reports?topic=&start=&end=` |
| گزارش کامل | `GET /api/v1/reports/{report_id}` |
| یک جامعه + اعضا (حداکثر ۵۰۰) | `GET /api/v1/communities/{report_id}/{community_id}?method=hybrid` |

`method` فقط `hybrid` (پیش‌فرض، خروجی اصلی) یا `louvain` (مقایسه).

خلاصهٔ هر گزارش: `id`, `topic`, `timeframe`, `hybrid.echo_count`, `hybrid.modularity`, `louvain.*`.

### ۳) اجرای پایپلاین (ادمین)

| کار UI | API |
|---|---|
| کانفیگ فعلی موضوع/بازه | `GET /api/v1/pipeline/config` |
| شروع ران | `POST /api/v1/pipeline/runs` → `202` |
| لیست ران‌ها | `GET /api/v1/pipeline/runs` |
| وضعیت + `log_tail` | `GET /api/v1/pipeline/runs/{run_id}` |

بدنهٔ شروع:

```json
{
  "fetch": true,
  "detect": true,
  "topic_label": "جنگ",
  "topic_query": "",
  "start_date": "2026-01-01",
  "end_date": "2026-01-14",
  "days": 14,
  "slot_modes": ["daily"],
  "auth": "1",
  "include_secondary": false,
  "max_scan_docs": 0
}
```

حداقل یکی از `fetch` یا `detect` باید `true` باشد. همزمان فقط یک ران `queued`/`running`؛ دومی `409` است. وضعیت: `queued` → `running` → `done` | `failed`.

### ۴) ریویو انسانی (لیبل اکو)

| کار UI | API |
|---|---|
| ساخت دستهٔ نمونه | `POST /api/v1/review/samples` → `201` |
| لیست نمونه‌ها | `GET /api/v1/review/samples?batch_id=&method=&labeled=true` |
| یک نمونه | `GET /api/v1/review/samples/{sample_id}` |
| ثبت لیبل | `POST /api/v1/review/labels` |
| دقت/فراخوانی/F1 | `GET /api/v1/review/metrics?batch_id=` |

```json
{ "method": "both", "sample_size": 20, "min_size": 1, "seed": 42 }
```

```json
{ "labels": [{ "sample_id": "...", "human_label": 1, "notes": "" }] }
```

`human_label`: `1` = اکو، `0` = غیر اکو. `method`: `hybrid` | `louvain` | `both`.

### ۵) تغییر حزب / جابه‌جایی نود

| کار UI | API |
|---|---|
| لیست با فیلتر | `GET /api/v1/party-changes?node_id=&from_party=&to_party=&slot_mode=&topic=` |
| یک رویداد | `GET /api/v1/party-changes/{event_id}` |
| ثبت رویداد (بک‌اند پایپلاین) | `POST /api/v1/party-changes` |

فیلدهای رویداد: `event_id`, `node_id`, `from_party`, `to_party`, `previous_slot`, `current_slot`, `slot_mode`, `topic`, `detected_at` (+ `received_at` از سرور).

---

## فایل‌های نمونهٔ UI (اگر HTML استاتیک را هم می‌خواهی)

از ماشینی که پایپلاین را ران کرده کپی کن؛ `*.html` در git نیست.

| فایل | نقش |
|---|---|
| `timeline_template.html` | سورس layout/CSS/JS تایم‌لاین |
| `timeline_dashboard.html` | خروجی نهایی با دادهٔ embed |
| `dashboard_<mode>_<start>_to_<end>.html` | پوستهٔ یک اسلات |
| `hybrid_graph_<mode>_<start>_to_<end>.html` | گراف vis-network |
| `hybrid_graph_<mode>_<start>_to_<end>.json` | nodes/edges همان گراف برای رندر در React |
| `dashboard_*_legend.json` | رنگ/حزب/تعداد |
| `lib/bindings/utils.js` | تعامل گراف؛ بدون این iframe می‌شکند |

همه را از ریشهٔ پروژه سرو کن تا path نسبی درست بماند.

لازم نیست: `template.html` قدیمی، کل آرشیو `louvain_*`، `topics_10.csv`، `community_detection.py`.

---

## موضوعات (`GET /api/v1/topics`)

```json
{ "topics": [{ "key": "جنگ", "label": "جنگ", "query": "query-war", "has_data": true, "sources": ["pipeline_config"] }] }
```

---

## چک‌لیست اتصال

1. `make docker-up` و باز کردن `/docs`
2. Import کردن `/openapi.json` در Postman
3. اگر توکن دارید: Authorize در Swagger
4. برای تایم‌لاین: `topics` → `dashboards` → `dashboards/{id}` → `files/...`
5. گراف را از `graph.nodes` و `graph.edges` بکش (iframe از `files.hybrid_graph` فقط جایگزین است)
6. `partyFocus` را از API انتظار نداشته باش
