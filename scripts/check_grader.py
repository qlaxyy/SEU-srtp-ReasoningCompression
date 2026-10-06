"""Small synthetic parser/grader smoke test; no model or benchmark generation."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'third_party/rebalance'))
from utils.parser import extract_answer
from utils.grader import check_is_correct


def main():
    checks=[(r'The result is \boxed{7}.','7',True),
            (r'It equals \boxed{\frac{1}{2}}.','0.5',True),
            (r'\boxed{5}','7',False)]
    for text,gold,expected in checks:
        answer=extract_answer(text)
        actual=bool(check_is_correct(answer,gold))
        if actual!=expected:raise AssertionError((answer,gold,actual))
    print('Parser/grader smoke: 3/3 synthetic checks passed. No benchmark results.')


if __name__=='__main__':main()
