import re
import unicodedata
from sudachipy import Dictionary, SplitMode


class VNTextPostProcessor:
    def __init__(self):
        self.tokenizer = Dictionary(dict="core").create()

    def normalize_line(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)

        punct_map = str.maketrans({
            ",": "、",
            ".": "。",
            "!": "！",
            "?": "？",
            "(": "（",
            ")": "）",
            "[": "【",
            "]": "】",
        })
        text = text.translate(punct_map)

        text = re.sub(r"[・･\.]{2,}", "……", text)
        text = re.sub(r"[ー―─‐-]{2,}", "――", text)
        text = re.sub(
            r"(?<=[\u3040-\u30ff\u4e00-\u9faf\u3000-\u303f])\s+(?=[\u3040-\u30ff\u4e00-\u9faf\u3000-\u303f])",
            "",
            text,
        )
        return text.strip()

    def process_box(self, lines: list[str]) -> dict:
        cleaned_lines = [self.normalize_line(line) for line in lines if line.strip()]
        full_text = "".join(cleaned_lines)

        # 1. Extract leading speaker tag like 【李空】
        speaker = ""
        speaker_match = re.match(r"^【([^】]{1,15})】(.*)$", full_text)
        if speaker_match:
            speaker = speaker_match.group(1).strip()
            dialogue = speaker_match.group(2).strip()
        else:
            dialogue = full_text

        # 2. Strip outer Japanese quotation brackets so Sugoi V4 translates clean prose
        dialogue = re.sub(r"^[「『（]+|[」』）]+$", "", dialogue).strip()

        # 3. Replace any remaining internal brackets that cause <unk> -> ?? in Sugoi V4
        dialogue = dialogue.replace("【", "").replace("】", "")

        tokens = self.tokenizer.tokenize(dialogue, SplitMode.C)
        oov_tokens = [m.surface() for m in tokens if m.is_oov()]

        return {
            "speaker": speaker,
            "dialogue": dialogue,       # Feed ONLY this into Sugoi V4
            "text": full_text,          # Keep full 【李空】「知ってる」 for the top Japanese line
            "lines": cleaned_lines,
            "oov_suspects": oov_tokens,
        }