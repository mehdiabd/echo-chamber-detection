# پنل تحلیلی نمای احزاب

دادهٔ پنل کنار گراف از همان endpoint جزئیات اسلات می‌آید.

`GET /api/v1/dashboards/{slot_id}`

پایهٔ لوکال: `http://localhost:8765`  
نمونه: `GET http://localhost:8765/api/v1/dashboards/daily_260101_to_260102`

---

## کجا بخوانی

`partyFocus` هم‌سطح `graph` است. گراف را از `graph.nodes` / `graph.edges` بکش و پنل را از `partyFocus.parties`.

لیست اسلات‌ها (`GET /api/v1/dashboards`) این فیلد را ندارد.

---

## شکل پاسخ

```json
{
  "id": "daily_260101_to_260102",
  "graph": {
    "node_count": 3,
    "edge_count": 2,
    "nodes": [],
    "edges": []
  },
  "partyFocus": {
    "parties": [
      {
        "name": "جریان زن‌زندگی‌آزادی",
        "stats": {
          "total": 438,
          "outgoing": 209,
          "incoming": 229
        },
        "top_interactions": [
          {"type": "نقل‌قول", "handle": "@CENTCOM", "count": 15},
          {"type": "ذکر", "handle": "@PMN_Amy", "count": 10},
          {"type": "ذکر", "handle": "@kokchanews", "count": 8}
        ]
      }
    ]
  }
}
```

`parties` بر اساس `stats.total` نزولی است. حداکثر ۶ مورد در `top_interactions`.

`type` معمولاً `نقل‌قول` یا `ذکر` است (گاهی `بازنشر` / `پاسخ` / `تعامل`).

---

## رندر UI

1. دراپ‌داون «نمای حزب»: `parties[].name` — پیش‌فرض اولین آیتم.
2. سه کارت حزب انتخاب‌شده: `stats.total` (کل تعامل)، `stats.outgoing` (خروجی)، `stats.incoming` (ورودی).
3. لیست `top_interactions`: راست `type` + `handle`، چپ `count`.
4. عرض نوار: `count / max(top_interactions.count)` برای همان حزب.

اگر `parties` خالی است پنل را نشان نده. آرتیفکت قدیمی بدون `partyFocus` داخل `hybrid_graph_*.json` مقدار `{"parties": []}` می‌دهد؛ پایپلاین را دوباره اجرا کن.
