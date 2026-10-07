"""Scan raw verified-run transcripts, returning one result per failed step."""
import json
from pathlib import Path
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.log import read_eval_log_sample
from ..logs import enable_zstd_zip
from inspect_scout import Result, Scanner, Transcript, scanner
from .memory import remembered_original

AUDIT = json.loads((Path(__file__).parent / 'data/audit_context.json').read_text())
PROVIDED = {'13.6', '62.1', '76.3'}


def failed_cases(t: Transcript) -> list[Transcript]:
    """Extract failed steps from complete model events and grading metadata."""
    scores = t.metadata.get('scores', {})
    if not scores and t.metadata.get('score_verify_scicode'):
        scores = {'verify_scicode': t.metadata['score_verify_scicode']}
    score = next((s for s in scores.values() if isinstance(s, dict) and isinstance(s.get('value'), dict)), None)
    if score is None:
        raise ValueError('Transcript does not include per-subproblem scores.')
    values = score['value']
    sample = t.metadata.get('sample_metadata', {})
    steps = [s['step_number'] for s in sample['sub_steps'] if s['step_number'] not in PROVIDED]
    model_events = [e for e in t.events if getattr(e, 'event', None) == 'model']
    if any(not e.input for e in model_events):
        # Older Scout readers leave pooled model inputs unresolved.
        enable_zstd_zip()
        sample = read_eval_log_sample(t.source_uri, uuid=t.transcript_id, resolve_attachments=True)
        model_events = [e for e in sample.events if getattr(e, 'event', None) == 'model']
    events = {}
    for e in model_events:
        if getattr(e, 'event', None) != 'model' or not e.output.choices:
            continue
        # A retried request replaces the earlier response to the same prompt.
        key = json.dumps([m.model_dump(mode='json', exclude_none=True) for m in e.input], sort_keys=True)
        events[key] = e
    events = list(events.values())
    if len(events) != len(steps):
        raise ValueError(f'{t.transcript_id}: {len(events)} requests for {len(steps)} steps')
    cases = []
    for index, (step, event) in enumerate(zip(steps, events)):
        if values[step] != 0:
            continue
        if step not in AUDIT:
            raise ValueError(f'Audit context does not cover {step}; rebuild it for this dataset.')
        context = dict(AUDIT[step])
        context['model'] = t.model
        context['grading'] = (score.get('metadata', {}).get('per_environment', {})).get(step, {})
        messages = [ChatMessageSystem(content='ANALYST REVIEW CONTEXT\n' + json.dumps(context))]
        messages.extend(event.input)
        choice = event.output.choices[0]
        messages.append(choice.message)
        cases.append(Transcript(
            transcript_id=f'{t.transcript_id}-{step}', task_id=step, model=t.model,
            score=0, success=False, messages=messages,
            metadata={'subproblem':step, 'empty_token_limit':not choice.message.text.strip() and choice.stop_reason=='max_tokens'},
        ))
    return cases


@scanner(events=['model'], version=5)
def verified_memory() -> Scanner[Transcript]:
    judge = remembered_original()
    async def scan(t: Transcript) -> list[Result]:
        if (t.model or '').endswith('/gpt-6-sol'):
            return [Result(label='excluded_model', value=False, answer='excluded_no_cot', explanation='GPT source transcripts are excluded because their CoT is unavailable.')]
        cases = failed_cases(t)
        if not cases:
            return [Result(label='no_failed_subproblems', value=False, answer='no_failed_subproblems', metadata={'classification':'no_failed_subproblems'}, explanation='All scored subproblems passed.')]
        results=[]
        for case in cases:
            result=await judge(case)
            result.label=case.metadata['subproblem']
            results.append(result)
        return results
    return scan
