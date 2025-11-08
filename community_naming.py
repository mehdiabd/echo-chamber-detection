"""ماژول نام‌گذاری هوشمند جوامع"""
import json
import networkx as nx
from collections import defaultdict


def analyze_community_content(center_node, neighbors, filepath="res.json"):
    """تحلیل محتوای تعاملات یک جامعه"""
    texts = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if data.get("sender") in [center_node] + neighbors:
                    if text := data.get("text"):
                        texts.append(text)
    except Exception as e:
        print(f"[content analysis error] {e}")
        
    return " ".join(texts).lower()


def get_community_type(text_content):
    """تشخیص نوع جامعه بر اساس محتوا"""
    topics = {
        "خبری": ["خبر", "گزارش", "اطلاع‌رسان", "پوشش", "انتشار"],
        "سیاسی": ["سیاست", "دولت", "مجلس", "انتخاب", "نظام"],
        "بین‌المللی": ["جهان", "بین‌الملل", "دیپلماسی", "خارج"],
        "اقتصادی": ["اقتصاد", "بازار", "تورم", "ارز", "قیمت"],
        "فرهنگی": ["فرهنگ", "هنر", "ادب", "سینما", "موسیق"],
        "اجتماعی": ["جامعه", "مردم", "شهروند", "اجتماع"],
        "حقوقی": ["حقوق", "قانون", "عدالت", "دادگاه"]
    }
    
    scores = {}
    for topic, keywords in topics.items():
        score = sum(text_content.count(kw) for kw in keywords)
        if score > 0:
            scores[topic] = score
            
    return max(scores.items(), key=lambda x: x[1])[0] if scores else None


def get_community_role(text_content):
    """تشخیص نقش/موضع جامعه"""
    roles = {
        "تحلیلگران": ["تحلیل", "بررسی", "مطالعه", "پژوهش"],
        "خبرنگاران": ["خبر", "گزارش", "پوشش", "رسان"],
        "فعالان": ["فعال", "کنشگر", "مدافع", "پیگیر"],
        "منتقدان": ["نقد", "انتقاد", "مخالف", "اعتراض"],
        "روشنفکران": ["روشنفکر", "متفکر", "اندیش", "نظر"],
        "نویسندگان": ["نویس", "نگار", "قلم", "مقاله"],
        "هنرمندان": ["هنر", "هنرمند", "خلاق", "آفرینش"]
    }
    
    scores = {}
    for role, keywords in roles.items():
        score = sum(text_content.count(kw) for kw in keywords)
        if score > 0:
            scores[role] = score
            
    return max(scores.items(), key=lambda x: x[1])[0] if scores else None


def name_community_from_center(center_node):
    """نام‌گذاری بر اساس کاربر مرکزی"""
    patterns = {
        "شبکه خبری": ["news", "خبر", "نیوز", "bbc", "cnn", "voa"],
        "رسانه برون‌مرزی": ["international", "world", "global", "برون"],
        "گروه سیاسی": ["polit", "سیاس", "حزب", "جناح"],
        "شبکه فرهنگی": ["art", "هنر", "cinema", "سینما", "موسیق"],
        "فعالان اجتماعی": ["social", "جامعه", "مردم", "اجتماع"],
        "تحلیلگران اقتصادی": ["econ", "اقتصاد", "بازار", "تجارت"],
        "فعالان حقوق بشر": ["rights", "حقوق", "عدالت", "democracy"]
    }
    
    node_lower = center_node.lower()
    for label, keywords in patterns.items():
        if any(kw in node_lower for kw in keywords):
            return label
            
    return None