"""
UrduHaluBench - fixed RAG prompts (identical instructions in English and Urdu).
Keep these unchanged for every model so results are comparable; report them in the paper appendix.
"""

ABSTAIN_EN = "The passage does not contain the answer."
ABSTAIN_UR = "متن میں اس سوال کا جواب موجود نہیں ہے۔"

SYSTEM = {
    "en": (
        "You are a question-answering assistant. Answer the question using ONLY the information "
        "in the given passage. Do not use any outside knowledge. "
        f'If the passage does not contain the answer, reply exactly: "{ABSTAIN_EN}" '
        "Keep the answer short (at most one sentence). Answer in English."
    ),
    "ur": (
        "آپ ایک سوال و جواب کے معاون ہیں۔ سوال کا جواب صرف دیے گئے متن میں موجود معلومات کی "
        "بنیاد پر دیں۔ کوئی بیرونی معلومات استعمال نہ کریں۔ "
        f"اگر متن میں جواب موجود نہیں ہے تو بالکل یہ لکھیں: \"{ABSTAIN_UR}\" "
        "جواب مختصر رکھیں (زیادہ سے زیادہ ایک جملہ)۔ جواب اردو میں دیں۔"
    ),
}

USER = {
    "en": "Passage:\n{context}\n\nQuestion: {question}\nAnswer:",
    "ur": "متن:\n{context}\n\nسوال: {question}\nجواب:",
}


def build_messages(language: str, context: str, question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM[language]},
        {"role": "user", "content": USER[language].format(context=context.strip(), question=question.strip())},
    ]
