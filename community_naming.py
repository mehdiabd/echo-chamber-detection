"""Shared community naming and classification helpers."""

import json
import re
from dataclasses import dataclass

COMMUNITY_LABELS = {
    "اصولگرا": "Accounts aligned with mainstream conservative and traditional political currents in Iran.",
    "پایداری": "Accounts associated with the ideological positions, narratives, and figures linked to the Paydari movement.",
    "اصلاح‌طلب": "Accounts aligned with reformist political discourse, reform-oriented politicians, and related media.",
    "سلطنت‌طلب": "Accounts supporting restoration of the Iranian monarchy or promoting monarchist political narratives.",
    "منافقین": "Accounts associated with or supportive of the MEK/MKO organization and its narratives.",
    "احمدی‌نژادی‌ها": "Accounts supporting Mahmoud Ahmadinejad, his political positions, or figures associated with his movement.",
    "ری‌استارتی‌ها": "Accounts associated with Restart ideology, narratives, leaders, or affiliated communities.",
    "عدالت‌خواه": "Accounts focused on anti-corruption, social justice, economic justice, and justice-oriented activism.",
    "فمنیست": "Accounts discussing women's rights, gender equality, feminist theories, and related activism.",
    "جدایی‌طلب‌های کرد": "Accounts promoting Kurdish separatist movements, Kurdish nationalism, or related political goals.",
    "جدایی‌طلب‌های ترک": "Accounts promoting Turkic/Azeri separatist movements, nationalism, or related political goals.",
    "سایر جدایی‌طلب‌ها": "Accounts supporting separatist or independence-oriented movements not covered by other specific categories.",
    "جریان زن‌زندگی‌آزادی": "Accounts primarily engaged with narratives, activism, and discourse related to the Woman, Life, Freedom movement.",
    "سوسیالیست (چپ‌گرا)": "Accounts aligned with left-wing socialist ideology, emphasizing anti-capitalist discourse, social ownership, labor rights, and progressive economic and political narratives.",
    "قشر خاکستری": "Accounts with mixed, unclear, non-ideological, or weakly defined political affiliation compared with other categories."
}

DEFAULT_COMMUNITY_LABEL = "قشر خاکستری"

COMMUNITY_LABEL_ALIASES = {
    "osoolgera": "اصولگرا",
    "osoolgar": "اصولگرا",
    "principlist": "اصولگرا",
    "اصول گرا": "اصولگرا",
    "paydari": "پایداری",
    "paaydari": "پایداری",
    "eslahtalab": "اصلاح‌طلب",
    "reformist": "اصلاح‌طلب",
    "اصلاح طلب": "اصلاح‌طلب",
    "saltanat": "سلطنت‌طلب",
    "monarchist": "سلطنت‌طلب",
    "سلطنت طلب": "سلطنت‌طلب",
    "monafegh": "منافقین",
    "منافق": "منافقین",
    "mko": "منافقین",
    "mek": "منافقین",
    "ahmadinezhad": "احمدی‌نژادی‌ها",
    "ahmadinejad": "احمدی‌نژادی‌ها",
    "احمدی‌نژادی": "احمدی‌نژادی‌ها",
    "احمدی نژادی": "احمدی‌نژادی‌ها",
    "restart": "ری‌استارتی‌ها",
    "ری استارتی": "ری‌استارتی‌ها",
    "ری استارتی‌ها": "ری‌استارتی‌ها",
    "ری‌استارتی": "ری‌استارتی‌ها",
    "edalatkhah": "عدالت‌خواه",
    "edalattalab": "عدالت‌خواه",
    "justice": "عدالت‌خواه",
    "عدالت طلب": "عدالت‌خواه",
    "عدالت‌طلب": "عدالت‌خواه",
    "feminist": "فمنیست",
    "feminism": "فمنیست",
    "barandaz": "جریان زن‌زندگی‌آزادی",
    "zan zendegi azadi": "جریان زن‌زندگی‌آزادی",
    "woman life freedom": "جریان زن‌زندگی‌آزادی",
    "زن زندگی آزادی": "جریان زن‌زندگی‌آزادی",
    "زن‌زندگی‌آزادی": "جریان زن‌زندگی‌آزادی",
    "kurd separatist": "جدایی‌طلب‌های کرد",
    "turk separatist": "جدایی‌طلب‌های ترک",
    "خاکستری": DEFAULT_COMMUNITY_LABEL,
    "قشر خاکستری": DEFAULT_COMMUNITY_LABEL,
    "unknown": DEFAULT_COMMUNITY_LABEL,
    "gray": DEFAULT_COMMUNITY_LABEL,
    "grey": DEFAULT_COMMUNITY_LABEL,
    "mixed": DEFAULT_COMMUNITY_LABEL,
}

COMMUNITY_LABEL_COLORS = {
    "اصولگرا": "#2563eb",              # آبی
    "پایداری": "#0f172a",              # سرمه‌ای
    "اصلاح‌طلب": "#16a34a",            # سبز
    "سلطنت‌طلب": "#facc15",            # زرد
    "منافقین": "#dc2626",              # قرمز
    "جریان زن‌زندگی‌آزادی": "#9333ea", # بنفش
    "ری‌استارتی‌ها": "#92400e",        # قهوه‌ای
    "احمدی‌نژادی‌ها": "#000000",       # سیاه
    "قشر خاکستری": "#9ca3af",          # خاکستری
    "عدالت‌خواه": "#ec4899",           # صورتی
}


def _normalize_label_key(value):
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = text.replace("\u200c", "")
    return re.sub(r"[\s_\-–—()\[\]{}«»\"'،,؛:/.]+", "", text)


_LABEL_LOOKUP = {
    _normalize_label_key(label): label
    for label in COMMUNITY_LABELS
}
_LABEL_LOOKUP.update({
    _normalize_label_key(alias): label
    for alias, label in COMMUNITY_LABEL_ALIASES.items()
})


def coerce_allowed_label(value, default=None):
    """Return the canonical allowed label for user/LLM text."""
    if value is None:
        return default

    if value in COMMUNITY_LABELS:
        return value

    text = str(value).strip()
    key = _normalize_label_key(text)
    if key in _LABEL_LOOKUP:
        return _LABEL_LOOKUP[key]

    for label in COMMUNITY_LABELS:
        if _normalize_label_key(label) in key:
            return label

    return default


def get_community_label_color(value, default_label=DEFAULT_COMMUNITY_LABEL):
    """Return the stable color assigned to a canonical community label."""
    label = coerce_allowed_label(value, default=default_label)
    return COMMUNITY_LABEL_COLORS.get(label, COMMUNITY_LABEL_COLORS[default_label])


def parse_community_classification_response(response, default=None):
    """Parse an LLM response and return a normalized classification dict."""
    if not response:
        label = coerce_allowed_label(default)
        return {
            "selected_label": label,
            "confidence": 0,
            "reasoning": "fallback"
        } if label else None

    payload = response
    if isinstance(response, str):
        text = response.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
        if not text.startswith("{"):
            match = re.search(r"\{.*\}", text, re.S)
            text = match.group(0) if match else text
        try:
            payload = json.loads(text)
        except Exception:
            payload = {"selected_label": response}

    if not isinstance(payload, dict):
        payload = {"selected_label": payload}

    label = coerce_allowed_label(payload.get("selected_label") or payload.get("label"), default)
    if not label:
        return None

    try:
        confidence = int(float(payload.get("confidence", 0)))
    except (TypeError, ValueError):
        confidence = 0
    confidence = max(0, min(100, confidence))

    return {
        "selected_label": label,
        "confidence": confidence,
        "reasoning": str(payload.get("reasoning") or payload.get("reason") or "").strip()
    }


@dataclass
class CommunityProfile:
    name: str
    confidence: int
    reasoning: str
    description: str
    members: list


def analyze_community_content(center_node, neighbors, filepath="res.json"):
    """تحلیل محتوای تعاملات یک جامعه"""
    texts = []
    members = set([center_node] + list(neighbors))
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read().strip()
            if raw.startswith("Total hits:"):
                raw = "\n".join(raw.splitlines()[1:]).strip()

        if raw.startswith("["):
            docs = json.loads(raw)
        else:
            docs = []
            for line in raw.splitlines():
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        for data in docs:
            if not isinstance(data, dict):
                continue
            username = (
                data.get("sender") or data.get("user_name") or
                data.get("username") or data.get("user_id")
            )
            if username in members:
                text = data.get("normalized_text") or data.get("content") or data.get("text")
                if text:
                    texts.append(text)
    except Exception as e:
        print(f"[content analysis error] {e}")
    return " ".join(texts).lower()


# New function for LLM-based community naming prompt
def build_community_classification_prompt(
    members,
    text_content,
    dominant_political_label=None,
):
    """Build a constrained classification prompt for LLM-based community naming."""

    labels_text = "\n".join(
        f"- {label}: {description}"
        for label, description in COMMUNITY_LABELS.items()
    )

    members_text = "\n".join(f"- {member}" for member in members)
    dominant_label_guidance = ""
    if dominant_political_label and dominant_political_label != DEFAULT_COMMUNITY_LABEL:
        dominant_label_guidance = f"""
        The members' dominant political metadata has already been normalized to:
        {dominant_political_label}

        Treat this as strong verified evidence. You MUST NOT select
        '{DEFAULT_COMMUNITY_LABEL}' when this normalized political label is available.
        """.strip()

    return f"""
        You are an expert political and social community classification system.

        Your task is NOT to generate a new community name.

        Your task is to select the SINGLE BEST MATCHING label from the provided list.

        You MUST choose exactly one label from the allowed labels.

        Choose the label that most accurately represents the dominant identity, interests, behavior, narratives, and content of the community.

        Do not create new labels.
        Do not combine labels.
        Do not suggest alternatives.

        If the community clearly aligns with one of the listed groups, select that group.
        Only select 'قشر خاکستری' when the community does not clearly align with any other category or exhibits mixed characteristics.

        Allowed Labels:
        {labels_text}

        Community Members:
        {members_text}

        Dominant Political Metadata:
        {dominant_label_guidance or "No specific normalized political label is available."}

        Community Content:
        {text_content}

        Return JSON only in the following format:
        {{
        "selected_label": "one of the allowed labels",
        "confidence": 0-100,
        "reasoning": "short explanation"
        }}
        """.strip()


def build_community_profile(members, text_content, llm_response):
    """
    Standard output object used by the entire project.
    """
    parsed = parse_community_classification_response(
        llm_response,
        default=DEFAULT_COMMUNITY_LABEL
    )
    if not parsed:
        parsed = {
            "selected_label": DEFAULT_COMMUNITY_LABEL,
            "confidence": 0,
            "reasoning": "No valid classification response."
        }

    return CommunityProfile(
        name=parsed["selected_label"],
        confidence=parsed["confidence"],
        reasoning=parsed["reasoning"],
        description=COMMUNITY_LABELS[parsed["selected_label"]],
        members=members
    )
