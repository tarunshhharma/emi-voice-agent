"""Compliance checks that run in code, whatever the model says or is told to say."""
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)

# India has no daylight saving, so a fixed offset is exact (and needs no tz database).
IST = timezone(timedelta(hours=5, minutes=30))
CALL_START_HOUR = 8  # 8 am inclusive
CALL_END_HOUR = 19  # 7 pm exclusive

# Spoken in place of a blocked sentence. Native script, like everything Priya says.
VERIFY_LINE = (
    "माफ़ कीजिए, जानकारी देने से पहले मुझे आपकी पहचान verify करनी होगी। "
    "कृपया अपना पूरा नाम और date of birth या phone number के आख़िरी चार digits बताइए।"
)
POLITE_LINE = "माफ़ कीजिए, मैं सिर्फ़ आपकी मदद के लिए call कर रही हूँ।"


# --- before the call: may we call this person at all? ---


def calling_hours_ok(now: datetime | None = None) -> bool:
    hour = (now or datetime.now(IST)).astimezone(IST).hour
    return CALL_START_HOUR <= hour < CALL_END_HOUR


def check_call_allowed(borrower, now: datetime | None = None) -> str | None:
    """Return None if the call may be placed, else the reason it must not be."""
    if borrower.dnd_flag:
        return "do_not_call"
    if not calling_hours_ok(now):
        return "outside_calling_hours"
    return None


# --- on every reply, before it is spoken ---


@dataclass
class GuardResult:
    allowed: bool
    text: str  # what may be spoken: the original, or a safe replacement
    reason: str | None = None


def _words(*alternatives: str) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(alternatives) + r")\b", re.IGNORECASE)


# English and Hinglish spellings. A keyword list is simple and testable but not exhaustive.
BLOCKED_PATTERNS = {
    "threat": _words(
        "legal action", "legal notice", "court", "police", "arrest(?:ed)?", "jail", "lawsuit",
        "kanooni", "giraftaar", "giraftar", "blacklist(?:ed)?", "case (?:kar|file)\\w*",
    ),
    "shaming": _words(
        "sharam", "sharm", "besharam", "badnaam", "beizzati", "izzat", "dhokebaaz",
        "fraudster", "cheater", "chor", "defaulter",
    ),
    "third_party_contact": _words(
        "family", "parivaar", "parivar", "ghar waal\\w*", "rishtedaar\\w*", "relatives?",
        "employer", "boss", "office mein", "neighbou?rs?", "padosi\\w*",
    ),
}

# Words that reveal a loan exists or what is owed. Banned until the caller is verified.
LOAN_WORDS = re.compile(
    r"\b(?:emi|loan|kist|karz|overdue|due|late fee|rs|rupees?|rupaye|rupaiye|rupay|"
    r"lakh|hazaa?r|thousand|hundred|sau)\b|₹",
    re.IGNORECASE,
)
MONTHS = _words(
    "january", "february", "march", "april", "june", "july", "august",
    "september", "sept", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
)
DATE_PATTERN = re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b")
DAY_AND_MONTH = re.compile(
    r"\b\d{1,2}(?:st|nd|rd|th)?\s*(?:of\s+)?(?:may|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)", re.IGNORECASE
)


# The same lists in Devanagari: Priya writes Hindi in native script for the Indian voice, so
# every Roman rule needs a Devanagari twin. Matching is on normalised text (see _plain), so
# nukta spellings (हज़ार / हजार) match, and at a word start, so suffixed forms (पुलिसवाले,
# घरवालों) match too. \b can't be used: Devanagari vowel signs aren't word characters.
_DEV = "\u0900-\u097F"


def _dev_words(*words: str) -> re.Pattern:
    return re.compile(f"(?<![{_DEV}])(?:" + "|".join(words) + ")")


def _plain(text: str) -> str:
    """Devanagari without nukta (ज़ -> ज, ड़ -> ड) and with ASCII digits, so variant spellings match."""
    decomposed = unicodedata.normalize("NFD", text).replace("\u093c", "")
    return decomposed.translate(str.maketrans("०१२३४५६७८९", "0123456789"))


BLOCKED_PATTERNS_DEVANAGARI = {
    "threat": _dev_words("पुलिस", "कोर्ट", "अदालत", "गिरफ्तार", "जेल", "कानून", "मुकदमा", "केस", "लीगल",
                         "ब्लैकलिस्ट"),
    "shaming": _dev_words("शर्म", "बेशर्म", "बदनाम", "बेइज्जत", "इज्जत", "धोखेबाज", "फ्रॉड", "चीटर", "चोर",
                          "डिफॉल्टर"),
    "third_party_contact": _dev_words("परिवार", "घरवाल", "घर वाल", "रिश्तेदार", "पडोस", "बॉस", "ऑफिस",
                                      "एम्प्लॉयर", "फैमिली"),
}
LOAN_WORDS_DEVANAGARI = _dev_words("ईएमआई", "लोन", "किस्त", "कर्ज", "ओवरड्यू", "ड्यू", "लेट फी", "रुप", "लाख",
                                   "हजार", "सौ")
MONTHS_DEVANAGARI = _dev_words("जनवरी", "फरवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर",
                               "सितम्बर", "अक्टूबर", "अक्तूबर", "नवंबर", "नवम्बर", "दिसंबर", "दिसम्बर")


def blocked_reason(text: str) -> str | None:
    """Which always-blocked category the text falls into (threat, shaming, third party), if any."""
    plain = _plain(text)
    for reason, pattern in BLOCKED_PATTERNS.items():
        if pattern.search(text) or BLOCKED_PATTERNS_DEVANAGARI[reason].search(plain):
            return reason
    return None


def _loan_numbers(loan) -> set[str]:
    values = {loan.principal, loan.emi_amount, loan.emi_amount + loan.late_fee}
    if loan.late_fee:
        values.add(loan.late_fee)
    return {str(v) for v in values}


def _discloses_loan(text: str, loan, echo: frozenset = frozenset()) -> bool:
    # Normalise first, so every check below sees ASCII digits (३५०० -> 3500) and nukta-free Devanagari.
    plain = re.sub(r"(?<=\d),(?=\d)", "", _plain(text))  # "25,000" -> "25000"
    if echo:  # words the caller just said, read back to them by verify_identity: not a disclosure
        plain = re.sub(r"[A-Za-z0-9]+", lambda m: " " if m.group(0).lower() in echo else m.group(0), plain)
    if LOAN_WORDS.search(plain) or MONTHS.search(plain):
        return True
    if LOAN_WORDS_DEVANAGARI.search(plain) or MONTHS_DEVANAGARI.search(plain):
        return True
    if DATE_PATTERN.search(plain) or DAY_AND_MONTH.search(plain):
        return True
    return any(re.search(rf"(?<!\d){n}(?!\d)", plain) for n in _loan_numbers(loan))


# Model output that isn't speech: tool-call JSON, markup, or a script the voice can't speak.
# Seen live: Priya said '{"name":"Chaman Lal","dob_or_last4":""}', then two Gujarati letters and '**Tool invocation failed'.
_NOT_SPEECH = re.compile(r'[{}\[\]<>|`\\]|\*\*|":|dob_or_last4|function_call|tool invocation', re.IGNORECASE)
_SPEAKABLE = re.compile("^[\x20-\x7E\u00A0\u0900-\u097F\u2010-\u2027\u20B9]*$")  # Roman, Devanagari, quotes, dashes, rupee


def looks_malformed(text: str) -> bool:
    return bool(_NOT_SPEECH.search(text)) or not _SPEAKABLE.match(text)


def check_reply(text: str, verified: bool, loan, echo: frozenset = frozenset()) -> GuardResult:
    """Check one piece of an agent reply. Returns a safe replacement if it must not be said
    (an empty replacement means: say nothing). echo: lower-case words of a verification
    read-back, i.e. the caller's own date or digits, which may be repeated back to them."""
    if looks_malformed(text):
        return GuardResult(False, "", "malformed_output")
    if reason := blocked_reason(text):
        return GuardResult(False, POLITE_LINE, reason)
    if not verified and _discloses_loan(text, loan, echo):
        return GuardResult(False, VERIFY_LINE, "loan_detail_before_verification")
    return GuardResult(True, text)


# --- logging ---


def mask_pii(text: str) -> str:
    """Hide phone numbers and dates (DOB, due dates) before text goes to the logs."""
    text = re.sub(r"\+?\d{10,13}", "[phone]", text)
    return DATE_PATTERN.sub("[date]", text)
