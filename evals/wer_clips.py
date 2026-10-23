"""The 20 test sentences for the STT comparison: what a borrower says on an EMI call.

tts:      text sent to the TTS voice. Hindi words in Devanagari, English words in Latin script,
          because that is how a Hinglish speaker actually pronounces them.
roman:    the same sentence in Roman script (reference for Roman-script STT output).
entities: the words that must come through for the agent to work (names, numbers, dates,
          negations). Each entity lists the spellings we accept, in any script.
Hindi numbers are spoken as words, the way callers say them.
"""

CLIPS = [
    {"id": "c01", "tts": "हाँ जी, मैं राहुल शर्मा बोल रहा हूँ।",
     "roman": "haan ji, main rahul sharma bol raha hoon",
     "entities": [["rahul sharma", "राहुल शर्मा"]]},
    {"id": "c02", "tts": "मेरी date of birth चौदह May उन्नीस सौ नब्बे है।",
     "roman": "meri date of birth chaudah may unnees sau nabbe hai",
     "entities": [["chaudah", "चौदह", "14"], ["may", "मई", "मे"], ["unnees sau nabbe", "उन्नीस सौ नब्बे", "1990"]]},
    {"id": "c03", "tts": "फ़ोन नंबर के last four digits तीन दो एक ज़ीरो हैं।",
     "roman": "phone number ke last four digits teen do ek zero hain",
     "entities": [["teen do ek zero", "तीन दो एक जीरो", "3210", "teen do ek shunya", "तीन दो एक शून्य", "तीन दो एक zero"]]},
    {"id": "c04", "tts": "मैं EMI परसों तक भर दूँगा, पक्का।",
     "roman": "main emi parson tak bhar dunga, pakka",
     "entities": [["parson", "parso", "parason", "परसों", "परसो"]]},
    {"id": "c05", "tts": "अभी पैसे नहीं हैं, मेरी job चली गई है।",
     "roman": "abhi paise nahi hain, meri job chali gayi hai",
     "entities": [["nahi", "नहीं", "nahin"], ["job", "जॉब", "जोब"]]},
    {"id": "c06", "tts": "मैंने पिछले हफ्ते ही payment कर दिया था।",
     "roman": "maine pichle hafte hi payment kar diya tha",
     "entities": [["payment", "पेमेंट"], ["pichle hafte", "pichhle hafte", "पिछले हफ्ते"]]},
    {"id": "c07", "tts": "मेरा UTR नंबर चार पाँच छह सात आठ है।",
     "roman": "mera utr number chaar paanch chhe saat aath hai",
     "entities": [["45678", "चार पांच छह सात आठ", "char panch chhah saat aath", "chaar paanch chhe saat aath", "chaar paanch chhah saat aath"]]},
    {"id": "c08", "tts": "Payment link WhatsApp पर भेज दीजिए।",
     "roman": "payment link whatsapp par bhej dijiye",
     "entities": [["whatsapp", "व्हाट्सएप", "व्हाट्सऐप", "वॉट्सऐप"]]},
    {"id": "c09", "tts": "यह amount गलत है, मेरा EMI इतना नहीं था।",
     "roman": "yeh amount galat hai, mera emi itna nahi tha",
     "entities": [["galat", "गलत"], ["nahi", "नहीं", "nahin"]]},
    {"id": "c10", "tts": "अभी मैं meeting में हूँ, कल सुबह ग्यारह बजे call कीजिए।",
     "roman": "abhi main meeting mein hoon, kal subah gyarah baje call kijiye",
     "entities": [["gyarah", "ग्यारह", "11"], ["subah", "सुबह"]]},
    {"id": "c11", "tts": "आप गलत नंबर पर call कर रहे हैं, मैं सुनीता हूँ।",
     "roman": "aap galat number par call kar rahe hain, main sunita hoon",
     "entities": [["galat number", "गलत नंबर", "गलत number", "wrong number"], ["sunita", "सुनीता"]]},
    {"id": "c12", "tts": "मैं branch manager बोल रहा हूँ, late fee waive कर दो।",
     "roman": "main branch manager bol raha hoon, late fee waive kar do",
     "entities": [["manager", "मैनेजर"], ["waive", "वेव", "वेव़"]]},
    {"id": "c13", "tts": "तीन हज़ार पाँच सौ रुपये मैं Friday को दे दूँगा।",
     "roman": "teen hazaar paanch sau rupaye main friday ko de dunga",
     "entities": [["3500", "teen hazaar paanch sau", "teen hazar panch sau", "तीन हजार पांच सौ"],
                  ["friday", "फ्राइडे", "शुक्रवार"]]},
    {"id": "c14", "tts": "मुझे थोड़ा time चाहिए, अगले महीने तक।",
     "roman": "mujhe thoda time chahiye, agle mahine tak",
     "entities": [["agle mahine", "अगले महीने"]]},
    {"id": "c15", "tts": "Please मुझे दोबारा call मत करना।",
     "roman": "please mujhe dobara call mat karna",
     "entities": [["mat karna", "मत करना"]]},
    {"id": "c16", "tts": "मेरा नाम प्रिया अय्यर है, date of birth तीन March उन्नीस सौ पचासी।",
     "roman": "mera naam priya iyer hai, date of birth teen march unnees sau pachaasi",
     "entities": [["priya iyer", "priya ayyar", "प्रिया अय्यर", "प्रिया अयर"], ["march", "मार्च"],
                  ["unnees sau pachaasi", "उन्नीस सौ पचासी", "1985"]]},
    {"id": "c17", "tts": "मैं अभी गाड़ी चला रहा हूँ, बाद में बात करते हैं।",
     "roman": "main abhi gaadi chala raha hoon, baad mein baat karte hain",
     "entities": [["baad mein", "बाद में", "baad me"]]},
    {"id": "c18", "tts": "क्या आप बता सकती हैं कि late fee कितनी है?",
     "roman": "kya aap bata sakti hain ki late fee kitni hai",
     "entities": [["late fee", "लेट फी", "लेट फ़ी"], ["kitni", "कितनी"]]},
    {"id": "c19", "tts": "मैं मोहम्मद आरिफ़ बोल रहा हूँ, लखनऊ से।",
     "roman": "main mohammed arif bol raha hoon, lucknow se",
     "entities": [["mohammed arif", "mohammad arif", "मोहम्मद आरिफ"], ["lucknow", "लखनऊ"]]},
    {"id": "c20", "tts": "ठीक है, आप मुझे SMS पर link भेज दो, मैं आज ही pay कर दूँगा।",
     "roman": "theek hai, aap mujhe sms par link bhej do, main aaj hi pay kar dunga",
     "entities": [["sms", "एसएमएस"], ["aaj hi", "आज ही"]]},
]

# Where Tarun's real recording differs from the script (misreads, not STT errors). Real clips
# are scored against what was actually said, so a reading slip doesn't count against a provider.
REAL_AS_SPOKEN = {
    "c04": {"tts": "मैं EMI परसों भर दूँगा, पक्का।",
            "roman": "main emi parson bhar dunga, pakka"},
    "c09": {"tts": "यह amount गलत है, मेरी EMI इतना नहीं है।",
            "roman": "yeh amount galat hai, meri emi itna nahi hai"},
    "c16": {"tts": "मेरा नाम प्रिया अय्यर है, date of birth तीन March उन्नीस सौ पच्चीस।",
            "roman": "mera naam priya iyer hai, date of birth teen march unnees sau pachees",
            "entities": [["priya iyer", "priya ayyar", "प्रिया अय्यर", "प्रिया अयर"], ["march", "मार्च"],
                         ["unnees sau pachees", "unnees sau pachchees", "उन्नीस सौ पच्चीस", "1925"]]},
}


def clip_for(condition: str, clip: dict) -> dict:
    """The reference to score against: the script, or what was actually said on a real clip."""
    if condition == "real":
        return {**clip, **REAL_AS_SPOKEN.get(clip["id"], {})}
    return clip
