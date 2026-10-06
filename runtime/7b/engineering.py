"""Small synthetic teacher-forced fixtures; excluded from benchmark scoring."""
import hashlib


def examples(tokenizer):
    rows = []
    for i, (question, reasoning, answer) in enumerate([
        ('What is 3 + 4?', 'Add the two integers carefully. Three plus four equals seven.\n\nThe sum has been determined and can be stated directly.', 'The result of the addition is seven, so the final answer is \\boxed{7}.'),
        ('What is the area of a rectangle with width 3 and height 5?', 'The area of a rectangle equals its width times its height.\n\nMultiplying three by five gives fifteen square units.', 'Using width times height, the area of this rectangle is \\boxed{15}.'),
    ]):
        tokens = tokenizer.encode(reasoning, add_special_tokens=False) + [151649]
        body = tokenizer.encode(answer, add_special_tokens=False)
        if len(tokens) < 16 or len(body) < 12:
            raise ValueError('Engineering fixture too short for replay checkpoints')
        rows.append(dict(dataset_index=i, source='synthetic-engineering', problem=question,
                         problem_sha256=hashlib.sha256(question.encode()).hexdigest(),
                         reasoning_ids=tokens, plain_body_ids=body))
    return rows
