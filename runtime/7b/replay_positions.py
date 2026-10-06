"""Replay only known historical answer positions; observe each new input once."""
import numpy as np
from positions import LivePositions


def historical_mask(tokens,prompt_length,cutoff):
    if not 1<=prompt_length<=len(tokens) or not 0<=cutoff<=len(tokens):raise ValueError('Invalid replay clock')
    phase=False;mask=np.zeros(len(tokens),dtype=np.bool_)
    for i in range(prompt_length,len(tokens)):
        if tokens[i]==151648:phase=False
        elif tokens[i]==151649:phase=True
        mask[i]=i<cutoff and phase and tokens[i]!=151643
    return mask


class ReplayPositions(LivePositions):
    def resume(self,rid,prompt_length,tokens,cutoff,replays):
        slot=self.add(rid,prompt_length)
        if len(tokens)>prompt_length and cutoff!=len(tokens)-1:raise ValueError('Replay must leave exactly one new prediction input')
        row=self.requests[rid]
        row.update(replay_prefill=len(tokens),history_cutoff=cutoff,history_mask=historical_mask(tokens,prompt_length,cutoff),replays=replays)
        return slot

    def finish(self,rid):
        row=super().finish(rid);row.pop('history_mask',None)
        return row

    def schedule(self,ids,starts,counts,prompt_lengths):
        if not (len(ids)==len(starts)==len(counts)==len(prompt_lengths)):raise ValueError('Scheduler geometry differs')
        if len(set(ids))!=len(ids) or any(int(n)<1 for n in counts) or sum(map(int,counts))>self.max_tokens:
            raise ValueError('Invalid schedule size')
        rows=np.zeros(self.capacity,dtype=np.int64);valid=np.zeros(self.capacity,dtype=np.bool_);generated=valid.copy()
        slots=np.zeros(self.max_tokens,dtype=np.int64);prediction=np.zeros(self.max_tokens,dtype=np.bool_)
        history=np.zeros(self.max_tokens,dtype=np.bool_);updates=[];offset=0
        for rid,start,count,prefill in zip(ids,starts,counts,prompt_lengths):
            row=self.requests[rid];start,count,prefill=int(start),int(count),int(prefill)
            original=row['prompt_length'];expected_prefill=row.get('replay_prefill',original)
            if start!=row['next_position'] or prefill!=expected_prefill:raise ValueError('Untracked replay or changed prompt')
            if start>=prefill and count!=1:raise ValueError('Multiple decode tokens unsupported')
            if start<prefill<start+count:raise ValueError('Mixed prefill/decode unsupported')
            slot=row['slot'];last=offset+count-1;absolute_last=start+count-1
            slots[offset:offset+count]=slot
            if 'history_mask' in row:
                stop=min(start+count,len(row['history_mask']))
                if stop>start:history[offset:offset+stop-start]=row['history_mask'][start:stop]
            # Old recomputed inputs must not update counters/query/phase again.
            if start+count>=prefill:
                if absolute_last<row.get('history_cutoff',0):raise ValueError('Prediction behind restored phase')
                rows[slot]=last;valid[slot]=True;generated[slot]=absolute_last>=original;prediction[last]=True
            updates.append((row,count));offset+=count
        for row,count in updates:row['next_position']+=count
        return dict(row_indices=rows,row_valid=valid,generated_input=generated,token_slots=slots,
            prediction_mask=prediction,scheduled_tokens=offset,history_answer_mask=history)
