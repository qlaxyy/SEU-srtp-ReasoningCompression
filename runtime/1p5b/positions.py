"""One causal prediction row per request; no future text or scheduler-row identity."""
import numpy as np


class LivePositions:
    def __init__(self, capacity=256, max_tokens=32768):
        self.capacity, self.max_tokens = capacity, max_tokens
        self.requests = {}
        self.free = list(reversed(range(capacity)))

    def add(self, rid, prompt_length):
        if rid in self.requests or prompt_length < 1 or not self.free:
            raise ValueError('Duplicate request, empty prompt or full request table')
        slot = self.free.pop()
        self.requests[rid] = dict(slot=slot, prompt_length=int(prompt_length), next_position=0)
        return slot

    def finish(self, rid):
        row = self.requests.pop(rid)
        self.free.append(row['slot'])
        return row

    def schedule(self, ids, starts, counts, prompt_lengths):
        if not (len(ids) == len(starts) == len(counts) == len(prompt_lengths)):
            raise ValueError('Scheduler geometry differs')
        if len(ids) != len(set(ids)) or any(int(n) < 1 for n in counts):
            raise ValueError('Duplicate request or nonpositive token count')
        if sum(map(int, counts)) > self.max_tokens:
            raise ValueError('Scheduled tokens exceed frozen capacity')
        # Slots are owned by requests, independent of vLLM's current row order.
        row_indices = np.zeros(self.capacity, dtype=np.int64)
        row_valid = np.zeros(self.capacity, dtype=np.bool_)
        generated_input = np.zeros(self.capacity, dtype=np.bool_)
        token_slots = np.zeros(self.max_tokens, dtype=np.int64)
        prediction_mask = np.zeros(self.max_tokens, dtype=np.bool_)
        updates, offset = [], 0
        for rid, start, count, prompt in zip(ids, starts, counts, prompt_lengths):
            if rid not in self.requests:
                raise ValueError('Unregistered request')
            row = self.requests[rid]
            start, count, prompt = int(start), int(count), int(prompt)
            if start != row['next_position'] or prompt != row['prompt_length']:
                raise ValueError('Replay/preemption, changed prompt, or out-of-order input')
            if start >= prompt and count != 1:
                raise ValueError('Speculation/multiple decode tokens unsupported')
            if start < prompt < start+count:
                raise ValueError('Mixed prompt and decode batch unsupported')
            slot, last = row['slot'], offset+count-1
            token_slots[offset:offset+count] = slot
            if start+count >= prompt:
                row_indices[slot], row_valid[slot] = last, True
                generated_input[slot] = start >= prompt
                prediction_mask[last] = True
            updates.append((row, count))
            offset += count
        for row, count in updates:
            row['next_position'] += count
        return dict(row_indices=row_indices, row_valid=row_valid,
                    generated_input=generated_input, token_slots=token_slots,
                    prediction_mask=prediction_mask, scheduled_tokens=offset)
