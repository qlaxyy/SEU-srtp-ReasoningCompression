"""
Self-contained answer extraction and correctness helpers for ASC evaluation.

Copy this file together with eval_asc_paper.py into a fresh environment.
It does not depend on data_processing/ or eval/.
"""

from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Any


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def strip_string(value: Any) -> str:
    string = str(value).strip()
    string = string.replace("\n", "")
    string = string.rstrip(".")
    string = string.replace("\\!", "")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("cfrac", "frac")
    string = string.replace("\\,", "")
    string = string.replace("\\:", "")
    string = string.replace("\\;", "")
    string = string.replace("\\ ", "")
    string = string.replace("\\$", "").replace("$", "")
    string = string.replace("\\%", "%").replace(r"\%", "%")
    string = string.replace("\\cdot", "")
    string = string.replace("\\mathbf", "").replace("\\mathrm", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    without_trailing_text = re.sub(r"\\text\{.*?\}$", "", string).strip()
    if without_trailing_text:
        string = without_trailing_text
    string = re.sub(r"\\text\{([^{}]*)\}", r"\1", string)
    string = re.sub(r"\\mbox\{.*?\}", "", string)
    string = re.sub(r"\{(c|m)?m\}(\^(2|3))?", "", string).strip()
    string = re.sub(r"p\.m\.$", "", string).strip()
    string = re.sub(r"(\d)\s*t$", r"\1", string).strip()
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")
    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)
    if string.startswith("."):
        string = "0" + string
    string = _fix_sqrt(string)
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)
    string = string.replace(" ", "")
    string = re.sub(r"(\\|,|\.)+$", "", string)
    return string


def _fix_fracs(string: str) -> str:
    parts = string.split("\\frac")
    if len(parts) == 1:
        return string
    out = parts[0]
    for part in parts[1:]:
        out += "\\frac"
        if part.startswith("{"):
            out += part
        elif len(part) >= 2:
            a, b = part[0], part[1]
            rest = part[2:]
            out += "{" + a + "}"
            out += b if b == "{" else "{" + b + "}"
            out += rest
        else:
            out += part
    return out


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        if "sqrt" not in a:
            int(a)
        if "sqrt" not in b:
            int(b)
        return f"\\frac{{{a}}}{{{b}}}"
    except Exception:
        return string


def _fix_sqrt(string: str) -> str:
    string = re.sub(r"\\sqrt(-?[0-9.a-zA-Z]+)", r"\\sqrt{\1}", string)
    string = re.sub(r"\\sqrt\s+(\w+)$", r"\\sqrt{\1}", string)
    return string


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract_boxed_answers(text: str) -> list[str]:
    answers = []
    for piece in re.split(r"\\?boxed\{", str(text))[1:]:
        depth = 0
        for i, ch in enumerate(piece):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    ans = piece[:i]
                    if i + 1 < len(piece) and piece[i + 1] == "%":
                        ans = piece[: i + 1]
                    answers.append(ans)
                    break
    return answers


def _clean_candidate(ans: str) -> str:
    ans = str(ans).strip().split("\n")[0]
    ans = ans.lstrip(":").rstrip(".").rstrip("/")
    return strip_string(ans)


def _is_plausible_answer(ans: str) -> bool:
    if not ans or len(ans) > 120:
        return False
    if not re.search(r"[0-9A-Za-z\\]", ans):
        return False
    noisy = [
        "step",
        "first",
        "second",
        "finally",
        "therefore",
        "hence",
        "however",
        "because",
        "since",
        "given",
        "assume",
        "consider",
    ]
    lower = ans.lower()
    return not any(word in lower for word in noisy)


def _number_pattern() -> str:
    return r"(?<!\d)-?\d[\d,]*(?:\.\d+)?(?!\d)"


def _find_numbers(text: str) -> list[str]:
    return re.findall(_number_pattern(), text)


def _latex_frac_pattern() -> str:
    return r"\\(?:dfrac|frac)\{[^{}]+\}\{[^{}]+\}"


def _plain_frac_pattern() -> str:
    return r"-?\d[\d,]*/\d[\d,]*"


def _complex_pattern() -> str:
    return r"-?\d+(?:\.\d+)?\s*[+-]\s*\d+(?:\.\d+)?i"


def _leading_answer_candidate(segment: str) -> str:
    segment = segment.lstrip("*: \r\n\t")
    segment = re.sub(r"^\\\[\s*", "", segment).lstrip()

    patterns = [
        _complex_pattern(),
        _latex_frac_pattern(),
        _plain_frac_pattern(),
        r"\*\*([A-Z][A-Za-z][A-Za-z .'-]*)\*\*",
        _number_pattern(),
        r"([A-Z][A-Za-z]+)",
    ]
    for pattern in patterns:
        match = re.match(pattern, segment)
        if not match:
            continue
        candidate = match.group(1) if match.lastindex else match.group(0)
        candidate = re.split(r"\s*(?:</think>|###|\*\*|\n|\.)", candidate, 1)[0]
        cleaned = _clean_candidate(candidate)
        if cleaned and _is_plausible_answer(cleaned):
            return cleaned
    return ""


def _extract_marked_numbers(text: str) -> list[str]:
    num_pat = _number_pattern()
    marker_pat = re.compile(
        r"\b(?:final\s+answer|the\s+answer|answer)\b\s*"
        r"(?:(?:is|are|would\s+be|should\s+be|:|=)\s*)?",
        flags=re.IGNORECASE,
    )
    candidates = []
    for match in marker_pat.finditer(text):
        segment = text[match.end():][:500]
        if not segment.strip():
            continue

        money = re.findall(r"\\?\$\s*(" + num_pat + r")", segment)
        if money:
            candidates.append(_clean_candidate(money[0]))
            continue

        segment_for_match = segment.lstrip("*: \r\n")
        leading = _leading_answer_candidate(segment_for_match)
        delayed_name = ""
        if leading and not re.fullmatch(r"[A-Za-z]+", leading):
            candidates.append(leading)
            continue
        if leading:
            delayed_name = leading

        verb_pat = re.compile(
            r"\b(?:take|takes|took|spend|spends|spent|cost|costs|earn|earns|make|makes|made|"
            r"profit|profits|receive|receives|need|needs|save|saves|saved|sleep|sleeps|slept|total|totals|"
            r"is|are|was|were|equals?)\b[^\d-]{0,80}(" + num_pat + r")",
            flags=re.IGNORECASE,
        )
        verb_match = verb_pat.search(segment_for_match)
        if verb_match:
            candidates.append(_clean_candidate(verb_match.group(1)))
            continue

        nums = re.findall(num_pat, segment_for_match)
        if nums:
            candidates.append(_clean_candidate(nums[0]))
            continue

        cleaned = _clean_candidate(segment_for_match)
        if _is_plausible_answer(cleaned):
            candidates.append(cleaned)
            continue
        if delayed_name:
            candidates.append(delayed_name)
    return [c for c in candidates if c]


def _extract_program_output(text: str) -> str:
    if "```output" not in text:
        return ""
    out = text.split("```output")[-1]
    if "```" in out:
        out = out.split("```", 1)[0]
    return _clean_candidate(out)


def _tail_candidate(text: str) -> str:
    tail = text[int(len(text) * 0.6):]

    final_phrase = _extract_tail_final_candidate(tail)
    if final_phrase:
        return final_phrase

    list_candidate = _extract_tail_list_candidate(tail)
    if list_candidate:
        return list_candidate

    frac_eq_matches = re.findall(
        r"=\s*(" + _latex_frac_pattern() + r"|" + _plain_frac_pattern() + r"|"
        + _complex_pattern() + r")",
        tail,
    )
    if frac_eq_matches:
        return _clean_candidate(frac_eq_matches[-1])

    eq_matches = re.findall(
        r"=\s*(" + _number_pattern() + r")\s*(?:$|\.|\n|,|\]|\)|\s)",
        tail,
    )
    if eq_matches:
        return _clean_candidate(eq_matches[-1])

    conclusion = re.findall(
        r"(?:therefore|so|hence|thus|finally)\s*,?\s*(.+?)(?:\n|$|\.\s*)",
        tail,
        flags=re.IGNORECASE,
    )
    if conclusion:
        nums = _find_numbers(conclusion[-1])
        if nums:
            return _clean_candidate(nums[-1])

    nums = _find_numbers(tail)
    if nums:
        front_nums = set(_find_numbers(text[: int(len(text) * 0.5)]))
        for num in reversed(nums):
            if num not in front_nums:
                return _clean_candidate(num)
        return _clean_candidate(nums[-1])

    nums = _find_numbers(text)
    return _clean_candidate(nums[-1]) if nums else ""


def _extract_tail_final_candidate(text: str) -> str:
    """Extract final-answer style phrases from compressed non-boxed outputs."""
    total_blocks = re.findall(
        r"(?:total|final\s+answer|answer|result)[\s\S]{0,260}",
        text,
        flags=re.IGNORECASE,
    )
    for block in reversed(total_blocks):
        eq_candidates = re.findall(
            r"=\s*("
            + _complex_pattern()
            + r"|"
            + _latex_frac_pattern()
            + r"|"
            + _plain_frac_pattern()
            + r"|"
            + _number_pattern()
            + r")",
            block,
        )
        if eq_candidates:
            return _clean_candidate(eq_candidates[-1])

    named_patterns = [
        r"(?:achieved\s+by|is\s+achieved\s+by)\s+\*\*([A-Z][A-Za-z][A-Za-z .'-]*)\*\*",
        r"(?:student|person|winner|choice)[^.\n]{0,120}?\s+is\s+\*\*([A-Z][A-Za-z][A-Za-z .'-]*)\*\*",
        r"(?:highest|greatest|largest|smallest|maximum|minimum)[^.\n]{0,120}?\s+"
        r"(?:is|are|by)\s+\*\*([A-Z][A-Za-z][A-Za-z .'-]*)\*\*",
        r"(?:achieved\s+by|is\s+achieved\s+by)\s+([A-Z][A-Za-z][A-Za-z .'-]*)",
        r"(?:student|person|winner|choice)[^.\n]{0,120}?\s+is\s+([A-Z][A-Za-z][A-Za-z .'-]*)",
        r"(?:highest|greatest|largest|smallest|maximum|minimum)[^.\n]{0,120}?\s+"
        r"(?:is|are|by)\s+([A-Z][A-Za-z][A-Za-z .'-]*)",
    ]
    for pattern in named_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            raw_candidate = matches[-1].strip()
            if not re.match(r"[A-Z]", raw_candidate):
                continue
            candidate = _clean_candidate(raw_candidate)
            if candidate:
                return candidate

    direct_patterns = [
        r"(?:total|final\s+answer|answer|result)[^.\n]{0,120}=\s*(.{0,80})",
        r"(?:final\s+answer\s+(?:should\s+be|is|would\s+be)|answer\s+(?:is|would\s+be)|"
        r"simplifies\s+to|is\s+indeed)\s+(.{0,120})",
    ]
    for pattern in direct_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not matches:
            continue
        segment = matches[-1]
        segment = re.split(r"(?:</think>|###|\n|\.)", segment, 1)[0]
        candidate = _leading_answer_candidate(segment)
        if candidate:
            return candidate
    return ""


def _extract_tail_list_candidate(text: str) -> str:
    """Extract final multi-number answers like 'the integers are -2 and 1'."""
    patterns = [
        r"(?:integers?|solutions?|values?|roots?|answers?)\s+"
        r"(?:are|is|would\s+be|should\s+be)\s+(.{0,180})",
        r"(?:thus|therefore|so|hence)[^.\n]{0,80}?\s+are\s+(.{0,180})",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not matches:
            continue
        segment = matches[-1].split("\n", 1)[0]
        segment = re.split(r"\.\s|</think>|###|\*\*", segment, 1)[0]
        nums = _find_numbers(segment)
        if len(nums) >= 2:
            return ",".join(_clean_candidate(num) for num in nums)
    return ""


def extract_all_answers(pred_str: str) -> list[str]:
    if not pred_str or not str(pred_str).strip():
        return []

    text = str(pred_str).strip()
    text = re.sub(r"<\|?\|?think\|?>", "", text, flags=re.IGNORECASE)

    boxed = [_clean_candidate(ans) for ans in extract_boxed_answers(text)]
    boxed = [ans for ans in boxed if ans and _is_plausible_answer(ans)]
    if boxed:
        if any(re.fullmatch(r"[A-Z]", ans) for ans in boxed):
            named = _extract_tail_final_candidate(text[int(len(text) * 0.5):])
            if named and named not in boxed:
                return boxed + [named]
        return boxed

    marked = _extract_marked_numbers(text)
    if marked:
        return marked

    program_output = _extract_program_output(text)
    if program_output:
        return [program_output]

    # If a model restarts after </think>, prefer the post-think section.
    sections = [text]
    if "</think>" in text:
        before, after = text.split("</think>", 1)
        sections = [after.strip(), before.strip(), text]

    for section in sections:
        if not section:
            continue
        candidate = _tail_candidate(section)
        if candidate:
            return [candidate]
    return []


def extract_answer(pred_str: str) -> str:
    answers = extract_all_answers(pred_str)
    return answers[-1] if answers else ""


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
def extract_ground_truth(dataset_name: str, item: dict[str, Any]) -> str:
    if dataset_name == "gsm8k":
        return strip_string(item.get("answer", ""))
    if dataset_name == "math":
        ans = str(item.get("answer", "")).strip()
        if ans:
            return strip_string(ans)
        solution = item.get("solution", "")
        return extract_answer(solution) if solution else ""
    for key in ["answer", "target", "gt", "label"]:
        if key in item:
            return strip_string(item[key])
    return ""


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
def _to_float(value: str) -> float | None:
    value = strip_string(value).replace(",", "")
    if not value:
        return None
    value = value.rstrip("%")

    frac_match = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", value)
    if frac_match:
        num = _to_float(frac_match.group(1))
        den = _to_float(frac_match.group(2))
        if num is not None and den not in (None, 0):
            return num / den

    plain_frac = re.fullmatch(r"(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)", value)
    if plain_frac:
        den = float(plain_frac.group(2))
        if den != 0:
            return float(plain_frac.group(1)) / den

    try:
        return float(value)
    except Exception:
        pass

    leading_number = re.fullmatch(r"(-?\d[\d,]*(?:\.\d+)?)[A-Za-z%]+.*", value)
    if leading_number:
        try:
            return float(leading_number.group(1).replace(",", ""))
        except Exception:
            return None
    return None


def _sympy_equal(pred: str, gt: str, prec: float) -> bool:
    try:
        from sympy import N, simplify
        from sympy.parsing.latex import parse_latex
        from sympy.parsing.sympy_parser import parse_expr
    except Exception:
        return False

    pred_norm = strip_string(pred)
    gt_norm = strip_string(gt)
    for parser in (parse_latex, parse_expr):
        try:
            p_expr = parser(pred_norm)
            g_expr = parser(gt_norm)
            if simplify(p_expr - g_expr) == 0:
                return True
            if abs(float(N(p_expr)) - float(N(g_expr))) < prec:
                return True
        except Exception:
            continue
    return False


def _assignment_rhs(value: str) -> str | None:
    match = re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*=([^=]+)", value)
    return match.group(1) if match else None


def _split_plain_answer_list(value: str) -> list[str] | None:
    if "," not in value:
        return None
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(?:\.\d+)?", value):
        return None
    if any(ch in value for ch in "()[]{}"):
        return None
    parts = [part for part in value.split(",") if part]
    return parts if len(parts) > 1 else None


def compare_answers(pred: Any, gt: Any, prec: float = 1e-3) -> bool:
    if pred is None or gt is None:
        return False

    if isinstance(pred, list) or isinstance(gt, list):
        pred_list = pred if isinstance(pred, list) else [pred]
        gt_list = gt if isinstance(gt, list) else [gt]
        matched_pred = set()
        matched_gt = set()
        for i, p in enumerate(pred_list):
            for j, g in enumerate(gt_list):
                if j in matched_gt:
                    continue
                if compare_answers(p, g, prec=prec):
                    matched_pred.add(i)
                    matched_gt.add(j)
                    break
        return len(matched_pred) == len(pred_list) and len(matched_gt) == len(gt_list)

    pred_s = strip_string(pred)
    gt_s = strip_string(gt)
    if not pred_s or not gt_s:
        return False
    if pred_s == gt_s:
        return True

    pred_rhs = _assignment_rhs(pred_s)
    gt_rhs = _assignment_rhs(gt_s)
    if pred_rhs is not None and gt_rhs is not None:
        return compare_answers(pred_rhs, gt_rhs, prec=prec)
    if pred_rhs is not None:
        return compare_answers(pred_rhs, gt_s, prec=prec)
    if gt_rhs is not None:
        return compare_answers(pred_s, gt_rhs, prec=prec)

    if "\\cup" in pred_s and "\\cup" in gt_s:
        return compare_answers(pred_s.split("\\cup"), gt_s.split("\\cup"), prec=prec)

    pred_list = _split_plain_answer_list(pred_s)
    gt_list = _split_plain_answer_list(gt_s)
    if pred_list is not None or gt_list is not None:
        return compare_answers(pred_list or [pred_s], gt_list or [gt_s], prec=prec)

    p_num = _to_float(pred_s)
    g_num = _to_float(gt_s)
    if p_num is not None and g_num is not None:
        if abs(p_num - g_num) < prec:
            return True
        if g_num != 0 and abs(p_num - g_num) / abs(g_num) < 1e-4:
            return True

    p_norm = pred_s.lower().replace(" ", "").rstrip(".%")
    g_norm = gt_s.lower().replace(" ", "").rstrip(".%")
    if p_norm == g_norm:
        return True

    return _sympy_equal(pred_s, gt_s, prec=prec)


def is_correct(item: dict[str, Any], pred_key: str = "prediction", prec: float = 1e-3) -> bool:
    copied = deepcopy(item)
    return compare_answers(copied.get(pred_key, ""), copied.get("answer", ""), prec=prec)
