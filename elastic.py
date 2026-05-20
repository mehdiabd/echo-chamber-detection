"""Module providing a flexible Elasticsearch query script."""
import os
import sys
import subprocess
import json
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from collections import Counter
import logging
import re

# Set up logging to both console and file
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# File handler for internal logs
file_handler = logging.FileHandler("debug.log", mode="w", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

# Console handler for system logs
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # Show warnings/errors from dependencies
console_formatter = logging.Formatter("%(levelname)s: %(message)s")
console_handler.setFormatter(console_formatter)

log.addHandler(file_handler)
log.addHandler(console_handler)

def extract_mentions(text):
    return re.findall(r"@(\w+)", text or "")

# Prompt the user to select the authentication type
# print("Select the Elasticsearch server (authentication type):")
# print("1. Production Elasticsearch server (API Key Authentication)")
# print("2. Temp Elasticsearch server (Basic Authentication)")
# auth_type = input("Enter 1 or 2: ").strip()

# Set variables based on the selected authentication type
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--auth", type=str, choices=["1", "2"], default="1", help="Auth type: 1=prod, 2=temp")
parser.add_argument(
    "--include-secondary",
    action="store_true",
    help="Run slow secondary political-tag enrichment from twitter_source.",
)
parser.add_argument(
    "--max-scan-docs",
    type=int,
    default=0,
    help="Limit docs processed in each primary scan (0 = unlimited).",
)
parser.add_argument(
    "--topic-query",
    type=str,
    default="",
    help="Override topic query for this run (accepted in custom operator format).",
)
parser.add_argument(
    "--start-date",
    type=str,
    default="",
    help="Override start date (YYYY-MM-DD).",
)
parser.add_argument(
    "--end-date",
    type=str,
    default="",
    help="Override end date (YYYY-MM-DD).",
)
args = parser.parse_args()
auth_type = args.auth

if auth_type == "1":
    # API Key Authentication
    # print("You selected Production Elasticsearch server (API Key Authentication).")
    cur_path = os.path.dirname(__file__)
    CERTIFICATE = os.path.join(cur_path, "ca.crt")
    ELASTICSEARCH_URL = "https://192.168.59.79:9200"
    AUTH = "YXYyeVRKWUJKSFpwMVdrTnZWRDc6UHhqRHBQa2ZUYW1yMnBwWTV3Ri0xUQ=="
    INDEX = "twitter_temp_data"

    # Create Elasticsearch client
    es = Elasticsearch(
        ELASTICSEARCH_URL,
        api_key=AUTH,
        ca_certs=CERTIFICATE,
        verify_certs=True,
        ssl_show_warn=False
    )
elif auth_type == "2":
    # Basic Authentication
    # print("You selected Temp Elasticsearch server (Basic Authentication).")
    cur_path = os.path.dirname(__file__)
    CERTIFICATE = os.path.join(cur_path, "http_ca.crt")
    ELASTICSEARCH_URL = "https://192.168.59.26:9200/"
    USERNAME = "m.abdolahi"
    PASSWORD = "@bd0l@h12345"
    INDEX = "twitter_maroufi"

    # Create Elasticsearch client
    es = Elasticsearch(
        ELASTICSEARCH_URL,
        basic_auth=(USERNAME, PASSWORD),
        verify_certs=True,
        ca_certs=CERTIFICATE,
        ssl_show_warn=False,
        ssl_assert_hostname=False
    )
else:
    # print("Invalid selection. Please run the script again and select 1 or 2.")
    sys.exit(1)

# Keyword list (operator semantics: * = OR, + = AND, - = NOT)
PROTEST_QUERY_RAW = """((('اعتراضات' * 'اعتراضی' * 'تجمعات' * 'تجمعها' * 'تجمعهای' * 'تجمع_ها' * 'تحصن' * 'تحصنات' * 'تظاهرات' * 'تجمع‌های' * 'معترضان' * 'معترضین' * 'اعتراضهای' * 'اعتراضها' * 'اعتراض‌های' * 'اعتراض‌ها' * 'اعتراض_های' * 'اعتراض_ها' * 'تجمع_های' * 'خیزش' * 'تجمع‌ها' * 'اعتصاب' * 'اعتصابات' * 'اعتصاب‌های' * 'اعتصاب_های' * 'اعتصابهای' * 'اعتصاب‌ها' * 'اعتصاب_ها' * 'اعتصابها' * 'اعتراض_سراسری' * 'اعتصابات_سراسری' * 'اعتراضات_سراسری' * 'تظاهرات_سراسری' * 'خیزش_سراسری') + ('کارگر' * 'کارگران' * 'کارگری' * 'بازار' * 'بازاری' * 'بازاریان' * 'معلم' * 'معلمان' * 'معلمی' * 'فرهنگیان' * 'بازنشسته' * 'بازنشستگان' * 'بازنشستگی' * 'دانشگاهیان' * 'دانشجویان' * 'دانشجو' * 'دانشگاه' * 'دانشجویی' * 'مالباخته' * 'مالباختگان' * 'پزشکان' * 'پرستاران' * 'آزمون' * 'خریداران' * 'کشته شدگان' * 'کشته_شدگان' * 'کشته‌شدگان' * 'معلمی' * 'حقوق' * 'مرغداران' * 'دامداران' * 'کشاورزان' * 'کشاورزی' * 'دامداری' * 'مرغداری' * 'مرغدار' * 'دامدار' * 'رانندگان' * 'راننده‌گان' * 'کامیوندارا' * 'کامیونداران' * 'کامیون‌دارا' * 'کامیون‌دارن' * 'کامیون_داران' * 'کامیون_دارا' * 'تبریز' * 'ارومیه' * 'اردبیل' * 'اصفهان' * 'جاوید_نام' * 'جاوید‌نام' * 'جاوید نام' * 'جاویدنام' * 'کرج' * 'ایلام' * 'بوشهر' * 'تهران' * 'شهرکرد' * 'بیرجند' * 'مشهد' * 'بجنورد' * 'اهواز' * 'زنجان' * 'سمنان' * 'زاهدان' * 'شیراز' * 'قزوین' * 'قم' * 'سنندج' * 'کرمان' * 'کرمانشاه' * 'یاسوج' * 'گرگان' * 'رشت' * 'خرم‌آباد' * 'ساری' * 'اراک' * 'بندرعباس' * 'همدان' * 'یزد' * 'ایران' * 'ایرانی' * 'مردم' * 'مردمی'))) - (('دهه_فجر' * 'دهه‌فجر' * 'فجر' * 'دهه فجر' * '22بهمن' * '22 بهمن' * '22_بهمن' * 'ترکیه' * 'سوریه' * 'عراق' * 'لبنان' * 'اسلواکی' * 'جمهوری چک' * 'پاکستان' * 'هند' * 'افغانستان' * 'طالبان' * 'تمرین' * 'تمرینات' * 'ورزش' * 'سلامتی' * 'چربی' * 'تحریر الشام' * 'تحریرالشام' * 'تحریر شام' * 'تحریرشام' * 'جولانی' * 'احمد شرع' * 'احمد_الشرع' * 'احمدالشرع' * 'احمد الشرع' * 'HTS' * 'علوی' * 'علویان' * 'سوری' * 'قسد' * 'اتریش' * 'اسراییل' * 'اوکراین' * 'ایتالیا' * 'آرژانتین' * 'آلمان' * 'آلودگی_هوا' * 'الودگی_هوا' * 'آلودگی' * 'آب و هوا' * 'تایوان' * 'چین' * 'روسیه' * 'عراق' * 'عربستان' * 'کره جنوبی' * 'کره شمالی' * 'هلند' * 'نمره' * 'امتحان' * 'صهیونیست' * 'صهیونیستی' * 'یمن' * 'یمنی' * 'اسراییلی' * 'اسرائیل' * 'اسرائیلی' * 'ایالت' * 'ایالات' * 'بحرین' * 'قطر' * 'دیپلماسی' * 'سفیر' * 'سفارت' * 'ورزشگاه' * 'فوتبال' * 'بازی' * 'پرسپولیس' * 'اخراج_گاریدو' * 'گاریدو' * 'استقلال' * 'مدیرعامل' * 'عربستان' * 'آمریکا' * 'هند' * 'ژاپن' * 'برزیل' * 'آلمان' * 'فرانسه' * 'بریتانیا' * 'ایتالیا' * 'کانادا' * 'استرالیا' * 'مکزیک' * 'ترکیه' * 'کره جنوبی' * 'کره_جنوبی' * 'کره‌جنوبی' * 'اسپانیا' * 'آفریقای جنوبی' * 'آرژانتین' * 'اندونزی' * 'آلمان' * 'آمریکا' * 'آنتیگوآ و باربودا' * 'آروژانتین' * 'استرالیا' * 'اتریش' * 'اتیوپی' * 'اردن' * 'اروگوئه' * 'اسپانیا' * 'اسواتینی' * 'اسلوانی' * 'اسلوواکی' * 'ایالات متحده آمریکا' * 'ایتالیا' * 'ایران' * 'ایسلند' * 'ایرلند' * 'برزیل' * 'بلاروس' * 'بلژیک' * 'بلیز' * 'بنگلادش' * 'بولیوی' * 'بوسنی' * 'بوتسوانا' * 'پاراگوئه' * 'پرو' * 'پرتغال' * 'پاکستان' * 'پاپوآ گینه نو' * 'تایلند' * 'تایوان' * 'ترکیه' * 'تونس' * 'تووالو' * 'سلیمان' * 'مارشال' * 'بریتانیا' * 'چاد' * 'چک' * 'دانمارک' * 'دومنیکا' * 'رومانی' * 'زامبیا' * 'زیمبابوه' * 'سرلانکا' * 'سربستان' * 'سیرالئون' * 'سیشل' * 'سنگاپور' * 'سوئد' * 'سوئیس' * 'سورینام' * 'سوریه' * 'سومالی' * 'سودان' * 'سنگال' * 'صربستان' * 'فرانسه' * 'فنلاند' * 'فلسطین' * 'فلیپین' * 'قبرس' * 'قرقیزستان' * 'قطر' * 'کامبوج' * 'کامرون' * 'کانادا' * 'قزاقستان' * 'کرواسی' * 'کوبا' * 'کویت' * 'کولومبیا' * 'کنیا' * 'کره جنوبی' * 'کره شمالی' * 'کوزوو' * 'کونگو' * 'کیریباتی' * 'لائوس' * 'لهستان' * 'لیبریا' * 'لیبی' * 'لیتوانی' * 'لیختن اشتاین' * 'لتونی' * 'لبنان' * 'لوکزامبورگ' * 'مالاوی' * 'مالدیو' * 'مالزی' * 'مالی' * 'مالت' * 'مراکش' * 'مکزیک' * 'مولدوا' * 'موناکو' * 'مونته نگرو' * 'موزامبیک' * 'نامیبیا' * 'نروژ' * 'نپال' * 'نیجر' * 'نیوزیلند' * 'نیکاراگوآ' * 'یمن' * 'یونان' * 'یمن' * 'فلسطین' * 'فلسطینی' * 'گردوخاک' * 'غزه' * 'گرد و خاک' * 'هواشناسی' * 'وزش باد' * 'ازمون_استخدامی' * 'ازمون‌استخدامی' * 'ازمون استخدامی' * ('نتایج' + 'امتحانات') * 'آزمون_استخدامی' * 'آزمون‌استخدامی' * 'آزمون استخدامی' * (('نتایج') + ('ازمون' * 'آزمون')) )) * ('اعتراض_سراسری' * 'اعتصابات_سراسری' * 'اعتراضات_سراسری' * 'تظاهرات_سراسری' * 'خیزش_سراسری' * 'iranmassacre' * 'digitalblackoutiran' * 'iranrevolution2026' * 'این_اخرین_نبرده_پهلوی_برمیگرده' * 'اعتراضات_سراسری' * 'r2pforiran' * 'digitalblackoutlran' * 'iranprotests' * 'iranianrevolution2026' * 'freeiran' * 'iranrevoiution2026') * ((اغتشاشات * 'نا_ارامی_ها' * 'نا_ارامی‌ها' * 'نا_ارامی ها' * 'نا_ارامیها' * 'نا‌ارامی_ها' * 'نا‌ارامی‌ها' * 'نا‌ارامی ها' * 'نا‌ارامیها' * 'نا ارامی_ها' * 'نا ارامی‌ها' * 'نا ارامی ها' * 'نا ارامیها' * 'ناارامی_ها' * 'ناارامی‌ها' * 'ناارامی ها' * 'ناارامیها' * 'اغتشاشگر' * 'اغتشاشگران' * اشوب * اشوبگر * 'اشوب_طلب' * 'اشوب‌طلب' * 'اشوب طلب' * 'معترضین ' * جانباختگان * 'جانباخته_های' * 'جانباخته‌های' * 'جانباخته های' * 'جانباختههای' * جانباخته * 'جاوید_نام' * 'جاوید‌نام' * 'جاوید نام' * 'جاویدنام' * 'شهید_راه_وطن' * 'شهید_راه‌وطن' * 'شهید_راه وطن' * 'شهید_راهوطن' * 'شهید‌راه_وطن' * 'شهید‌راه‌وطن' * 'شهید‌راه وطن' * 'شهید‌راهوطن' * 'شهید راه_وطن' * 'شهید راه‌وطن' * 'شهید راه وطن' * 'شهید راهوطن' * 'شهیدراه_وطن' * 'شهیدراه‌وطن' * 'شهیدراه وطن' * 'شهیدراهوطن' * 'شهید_راه_ازادی' * 'شهید_راه‌ازادی' * 'شهید_راه ازادی' * 'شهید_راهازادی' * 'شهید‌راه_ازادی' * 'شهید‌راه‌ازادی' * 'شهید‌راه ازادی' * 'شهید‌راهازادی' * 'شهید راه_ازادی' * 'شهید راه‌ازادی' * 'شهید راه ازادی' * 'شهید راهازادی' * 'شهیدراه_ازادی' * 'شهیدراه‌ازادی' * 'شهیدراه ازادی' * 'شهیدراهازادی' * 'شهید_امنیت' * 'شهید‌امنیت' * 'شهید امنیت' * 'شهیدامنیت' * 'مدافعان_امنیت' * 'مدافعان‌امنیت' * 'مدافعان امنیت' * 'معترضان' * 'کشته_شدگان' * 'کشته‌شدگان' * 'کشته شدگان' * 'کشته_شده' * 'کشته‌شده' * 'کشته شده' * 'کشته_شده_ها' * 'کشته_شده‌ها' * 'کشته_شده ها' * 'کشته_شدهها' * 'کشته‌شده_ها' * 'کشته‌شده‌ها' * 'کشته‌شده ها' * 'کشته‌شدهها' * 'کشته شده_ها' * 'کشته شده‌ها' * 'کشته شده ها' * 'کشته شدهها') + ('دی_ماه' * 'دی‌ماه' * 'دی ماه' * 'دیماه'))"""
PROTEST_QUERY_RAW = """((('اعتراضات' * 'اعتراضی' * 'تجمعات' * 'تجمعها' * 'تجمعهای' * 'تجمع_ها' * 'تحصن' * 'تحصنات' * 'تظاهرات' * 'تجمع‌های' * 'معترضان' * 'معترضین' * 'اعتراضهای' * 'اعتراضها' * 'اعتراض‌های' * 'اعتراض‌ها' * 'اعتراض_های' * 'اعتراض_ها' * 'تجمع_های' * 'خیزش' * 'تجمع‌ها' * 'اعتصاب' * 'اعتصابات' * 'اعتصاب‌های' * 'اعتصاب_های' * 'اعتصابهای' * 'اعتصاب‌ها' * 'اعتصاب_ها' * 'اعتصابها' * 'اعتراض_سراسری' * 'اعتصابات_سراسری' * 'اعتراضات_سراسری' * 'تظاهرات_سراسری' * '')))"""

if args.topic_query:
    PROTEST_QUERY_RAW = args.topic_query

def _to_query_string(raw: str) -> str:
    """Convert custom operators to Elasticsearch simple_query_string syntax."""
    # Use double quotes for phrases; Lucene expects "
    q = raw.replace("'", '"')
    # Convert custom operators with flexible spacing
    q = re.sub(r"\s*\*\s*", " OR ", q)      # OR
    q = re.sub(r"\s*\+\s*", " AND ", q)     # AND
    q = re.sub(r"\s*-\s*", " AND NOT ", q)  # NOT
    # Normalize spacing
    q = re.sub(r"\s+", " ", q).strip()
    return q


PROTEST_QUERY = _to_query_string(PROTEST_QUERY_RAW)

today = datetime.utcnow().date()
start_date = (today - timedelta(days=60)).strftime("%Y-%m-%d")
end_date = today.strftime("%Y-%m-%d")
if args.start_date:
    start_date = args.start_date
if args.end_date:
    end_date = args.end_date

query_body = {
    "track_total_hits": True,
    "query": {
        "bool": {
            "must": [
                {
                    "simple_query_string": {
                        "query": PROTEST_QUERY,
                        "fields": ["normalized_text", "text"],
                        "default_operator": "OR",
                        "flags": "OR|AND|NOT|PHRASE|PRECEDENCE",
                        "lenient": True
                    }
                }
            ],
            "filter": [
                {
                    "range": {
                        "date": {
                            "gte": start_date,
                            "lte": end_date,
                            "format": "yyyy-MM-dd"
                        }
                    }
                }
            ]
        }
    },
    "_source": [
        "user_name",
        "normalized_text",
        "content",
        "text",
        "comment",
        "keywords",
        "user_title",
        "political_category.label",
        "date",
        "timestamp",
        "crawl_date",
        "type",
        "reply_to_user",
        "reply_to",
        "in_reply_to_user",
        "in_reply_to_screen_name",
        "reply",
        "quote",
        "repost",
        "retweet",
        "retweeted_status",
        "quoted_status",
        "entity.mention",
        "entity.mention.user_name",
        "category.label",
        "emotion.label",
    ]
}

type_counter = Counter()

def _extract_user(obj):
    if not obj:
        return None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for key in ("user_name", "username", "screen_name", "user_id"):
            if obj.get(key):
                return obj.get(key)
        user = obj.get("user")
        if isinstance(user, dict):
            for key in ("user_name", "username", "screen_name", "user_id"):
                if user.get(key):
                    return user.get(key)
    return None

def _log_count(label, query):
    try:
        res = es.count(index=INDEX, body={"query": query})
        log.warning(f"[count] {label}: {res.get('count')}")
    except Exception as e:
        log.warning(f"[count] {label} failed: {e}")

# Diagnostics: check whether data exists in date range and whether query filters too hard
try:
    date_only_query = {"range": {"date": {"gte": start_date, "lte": end_date, "format": "yyyy-MM-dd"}}}
    _log_count("date_only", date_only_query)
    _log_count("query_only", query_body["query"]["bool"]["must"][0])
    _log_count("query_and_date", query_body["query"])
    # Index min/max date for alignment
    try:
        min_date = es.search(
            index=INDEX,
            body={"sort": [{"date": "asc"}], "_source": ["date"], "size": 1}
        )
        max_date = es.search(
            index=INDEX,
            body={"sort": [{"date": "desc"}], "_source": ["date"], "size": 1}
        )
        min_hit = (min_date.get("hits", {}).get("hits") or [{}])[0].get("_source", {})
        max_hit = (max_date.get("hits", {}).get("hits") or [{}])[0].get("_source", {})
        min_val = str(min_hit.get("date", ""))[:10]
        max_val = str(max_hit.get("date", ""))[:10]
        if min_val and max_val:
            log.warning(f"[date_range] index min={min_val} max={max_val}")
    except Exception as e:
        log.warning(f"[date_range] failed: {e}")
    # Simple term test to confirm field matches
    test_term = "ایران"
    test_query = {
        "simple_query_string": {
            "query": test_term,
            "fields": ["normalized_text", "text"],
            "default_operator": "OR",
            "flags": "OR|AND|NOT|PHRASE|PRECEDENCE",
            "lenient": True,
        }
    }
    _log_count("test_term_only", test_query)
    # Sample one doc to inspect available fields
    try:
        sample = es.search(index=INDEX, body={"query": {"match_all": {}}, "size": 1})
        hits = sample.get("hits", {}).get("hits", [])
        if hits:
            src = hits[0].get("_source", {})
            log.warning(f"[sample] keys: {sorted(src.keys())[:30]}")
            for k in ["normalized_text", "content", "text", "date", "type"]:
                if k in src:
                    val = src.get(k)
                    if isinstance(val, str):
                        log.warning(f"[sample] {k}: {val[:200]}")
                    else:
                        log.warning(f"[sample] {k}: {type(val).__name__}")
        else:
            log.warning("[sample] no hits for match_all")
    except Exception as e:
        log.warning(f"[sample] failed: {e}")
    # Sample docs with non-empty content-like fields
    for field in ["normalized_text", "text", "comment", "keywords", "content"]:
        try:
            sample_q = {"query": {"exists": {"field": field}}, "size": 1}
            sample = es.search(index=INDEX, body=sample_q)
            hits = sample.get("hits", {}).get("hits", [])
            if not hits:
                continue
            src = hits[0].get("_source", {})
            val = src.get(field)
            if isinstance(val, str) and val.strip():
                log.warning(f"[sample_nonempty] {field}: {val[:200]}")
                break
            if isinstance(val, list) and val:
                log.warning(f"[sample_nonempty] {field}: {str(val[:5])}")
                break
        except Exception:
            continue
except Exception as e:
    log.warning(f"[count] diagnostics skipped: {e}")

log.info("Starting initial scan query...")
primary_scan = scan(
    es,
    index=INDEX,
    query=query_body,
    preserve_order=True,
    scroll="10m"
)

# Extract all distinct usernames and displaynames
usernames = set()
displaynames = set()
COUNT = 0
try:
    with open("res.json", "w", encoding="utf-8") as f:
        for doc in primary_scan:
            if args.max_scan_docs and COUNT >= args.max_scan_docs:
                log.warning(f"[limit] Reached max-scan-docs={args.max_scan_docs} in primary scan.")
                break
            source = doc["_source"]
            raw_type = source.get("type")
            tweet_type = "quote" if raw_type in {"quote", "qoute"} else (raw_type or "").lower()
            if tweet_type in {"retweet", "re-tweet", "rt"}:
                tweet_type = "repost"
            if tweet_type not in {"post", "reply", "quote", "repost"}:
                log.warning(f"❓ Unrecognized tweet_type: {tweet_type} (raw={raw_type})")
            type_counter[tweet_type] += 1
            json.dump(source, f, ensure_ascii=False)
            f.write("\n")
            COUNT += 1
            uname = source.get("user_name")
            dname = source.get("user_title")
            if uname:
                usernames.add(uname)
            if dname:
                displaynames.add(dname)
finally:
    try:
        primary_scan.close()
    except Exception:
        pass
log.info(f"Total hits: {COUNT}")
log.info(f"TWEET TYPE COUNTS IN SCAN 1: {dict(type_counter)}")

# Rewind and re-scan to collect interactions
log.info("Starting scan to collect interactions...")
interaction_scan = scan(
    es,
    index=INDEX,
    query=query_body,
    preserve_order=True
)

written = 0
processed_docs = 0
EXAMPLE_LIMIT = 10
EXAMPLE_PRINTED = 0
try:
    with open("interactions.json", "w", encoding="utf-8") as f_interactions:
        for doc in interaction_scan:
            if args.max_scan_docs and processed_docs >= args.max_scan_docs:
                log.warning(f"[limit] Reached max-scan-docs={args.max_scan_docs} in interaction scan.")
                break
            processed_docs += 1
            source = doc["_source"]
            raw_type = source.get("type")
            tweet_type = "quote" if raw_type in {"quote", "qoute"} else (raw_type or "").lower()
            # Only warn for truly unrecognized tweet types
            if tweet_type not in {"post", "reply", "quote", "repost"}:
                log.warning(f"❓ Unrecognized tweet_type: {tweet_type} (raw={raw_type})")
            # Remove repetitive example doc logging
            # Remove repetitive info log for every doc
            # Remove debug log for every tweet type
            sender = source.get("user_name")
            date_val = source.get("date") or source.get("timestamp") or source.get("crawl_date")
            date_str = str(date_val)[:10] if date_val else ""

            if not sender or not tweet_type or not date_str:
                continue

            # Handle structured interaction types
            if tweet_type == "reply":
                target = (
                    source.get("reply_to_user")
                    or source.get("reply_to")
                    or source.get("in_reply_to_user")
                    or source.get("in_reply_to_screen_name")
                    or _extract_user(source.get("reply"))
                )
                if not target:
                    # Only warn once for missing target user
                    # log.warning("⛔ Skipped reply: No target user.")
                    continue
                elif target == sender:
                    # log.warning("⛔ Skipped reply: Target same as sender.")
                    continue
                else:
                    interaction = {
                        "sender": sender,
                        "target": target,
                        "type": "reply",
                        "date": date_str
                    }
                    log.info(f"Writing interaction: {interaction}")
                    json.dump(interaction, f_interactions, ensure_ascii=False)
                    f_interactions.write("\n")
                    written += 1
            elif tweet_type == "quote":
                quote = source.get("quote") or source.get("quoted_status") or {}
                if not isinstance(quote, dict):
                    # log.warning("⛔ Skipped quote: Not a dict.")
                    continue
                else:
                    quoted_user = _extract_user(quote)
                    if not quoted_user:
                        # log.warning("⛔ Skipped quote: No quoted user.")
                        continue
                    elif quoted_user == sender:
                        # log.warning("⛔ Skipped quote: Self-quote.")
                        continue
                    else:
                        interaction = {
                            "sender": sender,
                            "target": quoted_user,
                            "type": "quote",
                            "date": date_str
                        }
                        log.info(f"Writing interaction: {interaction}")
                        json.dump(interaction, f_interactions, ensure_ascii=False)
                        f_interactions.write("\n")
                        written += 1

            elif tweet_type == "repost":
                repost = source.get("repost") or source.get("retweeted_status") or source.get("retweet") or {}
                if not isinstance(repost, dict):
                    # log.warning("⛔ Skipped repost: Not a dict.")
                    continue
                else:
                    retweeted_user = _extract_user(repost)
                    if not retweeted_user:
                        # log.warning("⛔ Skipped repost: No retweeted user.")
                        continue
                    elif retweeted_user == sender:
                        # log.warning("⛔ Skipped repost: Self-repost.")
                        continue
                    else:
                        interaction = {
                            "sender": sender,
                            "target": retweeted_user,
                            "type": "repost",
                            "date": date_str
                        }
                        log.info(f"Writing interaction: {interaction}")
                        json.dump(interaction, f_interactions, ensure_ascii=False)
                        f_interactions.write("\n")
                        written += 1

            elif tweet_type == "post":
                # log.warning("🟡 Processing a POST tweet...")
                mentions = source.get("entity", {}).get("mention", [])
                normalized_mentions = []
                if isinstance(mentions, list):
                    for mention in mentions:
                        if isinstance(mention, str):
                            normalized_mentions.append(mention)
                        elif isinstance(mention, dict):
                            name = (
                                mention.get("user_name")
                                or mention.get("username")
                                or mention.get("screen_name")
                                or mention.get("name")
                            )
                            if name:
                                normalized_mentions.append(name)
                text_blob = source.get("normalized_text") or source.get("content") or ""
                if normalized_mentions:
                    mentions = normalized_mentions
                else:
                    mentions = extract_mentions(text_blob)
                    # log.warning(f"🔍 Mentions found after fallback: {mentions}")

                if mentions:
                    for mention in mentions:
                        target = mention
                        if target and target != sender:
                            interaction = {
                                "sender": sender,
                                "target": target,
                                "type": "mention",
                                "date": date_str
                            }
                            log.info(f"Writing interaction: {interaction}")
                            json.dump(interaction, f_interactions, ensure_ascii=False)
                            f_interactions.write("\n")
                            written += 1
                else:
                    # Only warn once for missing mentions
                    # log.warning("⛔ Skipped post: No mentions found (entity or text).")
                    continue
finally:
    try:
        interaction_scan.close()
    except Exception:
        pass

log.warning(f"✅ TOTAL INTERACTIONS WRITTEN: {written}")
log.info("Extracted interactions written to interactions.json")
log.info("Finished writing interactions.json.")
log.info("Finished processing all interaction types.")

if args.include_secondary:
    log.info("Querying twitter_source for political tags...")

    def chunked(iterable, size):
        iterable = list(iterable)
        return [iterable[i:i + size] for i in range(0, len(iterable), size)]

    log.info("Chunking user_name and user_title lists for political tagging...")
    username_chunks = chunked(usernames, 100)
    displayname_chunks = chunked(displaynames, 100)

    total_batches = len(username_chunks) + len(displayname_chunks)
    batch_index = 1
    secondary_count = 0

    with open("res.json", "w", encoding="utf-8") as f:
        for chunk in username_chunks:
            log.info(f"Querying batch {batch_index}/{total_batches} [usernames]")
            query = {
                "query": {
                    "bool": {
                        "should": [{"match": {"user_name": uname}} for uname in chunk],
                        "minimum_should_match": 1
                    }
                },
                "_source": [
                    "user_name",
                    "normalized_text",
                    "content",
                    "user_title",
                    "political_category.label",
                    "type",
                    "entity.mention",
                    "category.label",
                    "emotion.label",
                    "entity.hashtag",
                    "language.label",
                    "ner.event",
                    "ner.location",
                    "ner.nationality",
                    "ner.organ",
                    "ner.person",
                    "ner.politic_group",
                    "offensive.label",
                    "sentiment.label"
                ]
            }
            results = es.search(index="twitter_source", body=query, size=1000)["hits"]["hits"]
            for doc in results:
                source = doc["_source"]
                uname = source.get("user_name")
                dname = source.get("user_title")
                political = source.get("political_category", {})
                label = political.get("label") if isinstance(political, dict) else None
                if uname and dname and label:
                    output_doc = {
                        "user_name": uname,
                        "user_title": dname,
                        "political_label": label,
                        "normalized_text": source.get("normalized_text"),
                        "content": source.get("content"),
                        "category": source.get("category", {}).get("label"),
                        "emotion": source.get("emotion", {}).get("label"),
                        "hashtag": source.get("entity", {}).get("hashtag"),
                        "language": source.get("language", {}).get("label"),
                        "event": source.get("ner", {}).get("event"),
                        "location": source.get("ner", {}).get("location"),
                        "nationality": source.get("ner", {}).get("nationality"),
                        "organization": source.get("ner", {}).get("organ"),
                        "person": source.get("ner", {}).get("person"),
                        "politic_group": source.get("ner", {}).get("politic_group"),
                        "offensive": source.get("offensive", {}).get("label"),
                        "sentiment": source.get("sentiment", {}).get("label")
                    }
                    json.dump(output_doc, f, ensure_ascii=False)
                    f.write("\n")
                    secondary_count += 1
            batch_index += 1

        for chunk in displayname_chunks:
            log.info(f"Querying batch {batch_index}/{total_batches} [displaynames]")
            query = {
                "query": {
                    "bool": {
                        "should": [{"match": {"user_title": dname}} for dname in chunk],
                        "minimum_should_match": 1
                    }
                },
                "_source": [
                    "user_name",
                    "normalized_text",
                    "content",
                    "user_title",
                    "political_category.label",
                    "type",
                    "entity.mention",
                    "category.label",
                    "emotion.label",
                    "entity.hashtag",
                    "language.label",
                    "ner.event",
                    "ner.location",
                    "ner.nationality",
                    "ner.organ",
                    "ner.person",
                    "ner.politic_group",
                    "offensive.label",
                    "sentiment.label"
                ]
            }
            results = es.search(index="twitter_source", body=query, size=1000)["hits"]["hits"]
            for doc in results:
                source = doc["_source"]
                uname = source.get("user_name")
                dname = source.get("user_title")
                political = source.get("political_category", {})
                label = political.get("label") if isinstance(political, dict) else None
                if uname and dname and label:
                    output_doc = {
                        "user_name": uname,
                        "user_title": dname,
                        "political_label": label,
                        "normalized_text": source.get("normalized_text"),
                        "content": source.get("content"),
                        "category": source.get("category", {}).get("label"),
                        "emotion": source.get("emotion", {}).get("label"),
                        "hashtag": source.get("entity", {}).get("hashtag"),
                        "language": source.get("language", {}).get("label"),
                        "event": source.get("ner", {}).get("event"),
                        "location": source.get("ner", {}).get("location"),
                        "nationality": source.get("ner", {}).get("nationality"),
                        "organization": source.get("ner", {}).get("organ"),
                        "person": source.get("ner", {}).get("person"),
                        "politic_group": source.get("ner", {}).get("politic_group"),
                        "offensive": source.get("offensive", {}).get("label"),
                        "sentiment": source.get("sentiment", {}).get("label")
                    }
                    json.dump(output_doc, f, ensure_ascii=False)
                    f.write("\n")
                    secondary_count += 1
            batch_index += 1

    log.info(f"Political tag hits: {secondary_count}")
    log.info("Finished writing political tags to res.json.")

    # Automatically call normalize_json.py only when secondary enrichment is enabled.
    cur_path = os.path.dirname(__file__)
    normalize_script = os.path.join(cur_path, "normalize_json.py")
    subprocess.run(["python3", normalize_script], check=True)
    log.info("Normalization complete.")
else:
    log.warning("[fast] Skipping secondary enrichment. Use --include-secondary for full tagging.")

# Show preview of res.json (last 3 lines)
with open("res.json", "r", encoding="utf-8") as f:
    lines = f.readlines()
    preview = lines[-3:] if len(lines) >= 3 else lines
    print("\n--- Preview of last 3 lines of res.json ---")
    for line in preview:
        print(line.strip())

log.info("All steps completed successfully.")
log.info("Finished full pipeline.")
