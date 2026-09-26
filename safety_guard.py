import re
from typing import Tuple

CRISIS_PATTERNS = [
    r"\b(suicide|suicidal|end my life|kill myself|want to die|don't want to live)\b",
    r"\b(self-harm|cut myself|overdose|slit my wrist|hurt myself)\b",
    r"\b(no reason to live|better off dead|hang myself)\b"
]

CRISIS_RESPONSE = """⚠️ **Immediate Support Resources Available**

I hear that you are going through immense pain, but as an AI companion, I cannot provide crisis clinical counseling. **You do not have to carry this alone.** Please reach out directly to professionals who can help you safely right now:

- **Tele-MANAS (India):** Call `14416` or `1800-891-4416` (24/7 toll-free)
- **KIRAN Helpline (India):** `1800-599-0019`
- **988 Suicide & Crisis Lifeline (US & Canada):** Call or text `988`
- **Emergency Services:** Call `112` or `911` immediately

Please connect with someone you trust or a crisis responder right now. They are ready to listen."""

def check_crisis_intent(text: str) -> Tuple[bool, str]:
    lowered = text.lower()
    for pattern in CRISIS_PATTERNS:
        if re.search(pattern, lowered):
            return True, CRISIS_RESPONSE
    return False, ""