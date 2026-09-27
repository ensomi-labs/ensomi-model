"""Generate, reparse and gate scoped native evidence before semantic review.

Completion and numeric checks cannot promote a checkpoint. Semantic and musical
review remains required, with output identities and witness clocks retained.
"""
from dataclasses import asdict
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import psutil
import torch

from ...osu_core.difficulty import parse_osu_file,compute_mania_star_rating_20241007
from ..audio_memory_continuation.generation import MemorySession
from ..audio_memory_continuation.model import load_model as load_memory
from ..bounded_typed_continuation.memory import footprint_bytes
from ..controlled_audio_continuation.allocation import LnAmountFeedback
from ..controlled_audio_continuation.generation import ControlledSession
from ..controlled_audio_continuation.model import load_model
from ..controlled_audio_continuation.outcomes import scoped_difficulty
from ..joint_audio_continuation.data import digest
from ..joint_audio_continuation.generation import save_rollout
from ..oracle_time_continuation.data import source_rows
from ..planned_audio_continuation.session import PublicationLog,_BudgetStop
from ..player_response.envelope import AttackEnvelope
from ..source_action_modeling.actions import parse_source
from ..typed_audio_continuation.controls import ControlSchedule,ControlSpan
from ..typed_audio_continuation.program import Recovery
from .publication import publication_report
from .report import evaluate_scopes
from .review import pressure_review_contexts
from .temporal import ChartTrace,Scope
from .witnesses import sustained_attack_witnesses


def _write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def _verified(path,expected):
    if digest(path)!=expected:raise ValueError('Qualification input bytes differ: '+str(path))


def _plan(path):
    plan=json.loads(Path(path).read_text())
    if plan.get('format')!='native-gameplay-qualification/v1' or not plan.get('cases'):
        raise ValueError('Expected a nonempty native-gameplay-qualification/v1 plan')
    keys=set()
    for case in plan['cases']:
        key=case['key'];duration=case['asset']['duration_ms']
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*',key) or key in keys:
            raise ValueError('Qualification case keys must be unique safe directory names')
        keys.add(key)
        if type(duration) is not int or duration<=0:raise ValueError('Case duration must be positive native ms')
        scopes=[Scope(s['name'],s['start_ms'],s['end_ms']) for s in case['scopes']]
        if not scopes or len({s.name for s in scopes})!=len(scopes) or any(s.end_ms>duration+1 for s in scopes):
            raise ValueError('Cases require unique observed scopes')
        for span in case['controls']:ControlSpan(**span)
        if case.get('switch') is not None:
            if case.get('head_times_ms') is not None:
                raise ValueError('Live control changes require a native H planner')
            switch=case['switch'];span=ControlSpan(**switch['span'])
            if not 0<=switch['announce_after_ms']<span.start_ms<span.end_ms<=duration+1:
                raise ValueError('Live control scope must follow its announcement')
    return plan


@torch.inference_mode()
def _generate(model,session_type,case,settings,guard):
    asset=case['asset'];duration=asset['duration_ms']
    for name in ('audio','mel'):_verified(asset[name+'_file'],asset[name+'_sha256'])
    controls=ControlSchedule(tuple(ControlSpan(**s) for s in case['controls']),model.style_names)
    times=case.get('head_times_ms')
    if times is not None:
        if any(type(t) is not int or not 0<=t<=duration for t in times) or any(a>=b for a,b in zip(times,times[1:])):
            raise ValueError('Fixed-H diagnostics require strictly increasing native times')
        times=tuple(times)
    session=session_type(model,np.load(asset['mel_file']),duration,controls,seed=case['seed'],
        max_seconds=settings.max_case_seconds,head_times=times,
        ln_feedback=LnAmountFeedback() if settings.ln_feedback else None)
    publications=PublicationLog(session.started,duration,None)
    startup=None;services=[];prefix=None;switched=False;reason='complete'
    try:
        while session.coverage<duration:
            guard()
            end=min(duration,max(0,session.coverage)+settings.service_ms)
            if case.get('switch') is not None and not switched:end=min(end,case['switch']['announce_after_ms'])
            before=session.coverage;tick=time.perf_counter()
            while session.coverage<end:
                guard();single=time.perf_counter();update=session.step(stop_at=end)
                publications.publish(update,time.perf_counter()-single)
                if startup is None and publications.rows>=30 and session.coverage>=min(settings.lead_ms-1,duration):
                    startup=publications.coverage[-1]['elapsed_seconds']
            services.append(dict(before_ms=before,coverage_ms=session.coverage,seconds=time.perf_counter()-tick))
            if case.get('switch') is not None and not switched and session.coverage==case['switch']['announce_after_ms']:
                prefix=tuple(session.rows);session.update_controls(ControlSpan(**case['switch']['span']));switched=True
    except _BudgetStop as error:reason=str(error)
    if prefix is not None and tuple(session.rows[:len(prefix)])!=prefix:
        raise RuntimeError('Control change rewrote published rows')
    actual=tuple(r.time_ms for r in session.rows if any(a in (1,2) for a in r.actions))
    if times is not None and actual!=tuple(t for t in times if t<=session.coverage):
        raise RuntimeError('Fixed-H diagnostic changed its supplied skeleton')
    result=publications.result(session,reason,controls=[asdict(s) for s in session.controls.spans],
        recovery=asdict(model.recovery),ln_feedback=asdict(LnAmountFeedback()) if settings.ln_feedback else None,
        service_windows=services,operational_startup_seconds=startup,
        maximum_service_seconds=max((s['seconds'] for s in services),default=0.),
        head_source='native' if times is None else 'fixed-times',committed_prefix_preserved=True,
        sampling_contract='r1-release-window-v1',
        latency_origin='Loaded CPU model and cached Mel, including complete model audio encoding and actual publication.')
    deadlines=publication_report(publications.coverage,duration,lead_ms=settings.lead_ms,startup_seconds=startup)
    return result,deadlines,publications.coverage


def _inspect(case,saved,result,deadlines,settings,envelope,identity):
    path=Path(saved['osu_file']);osu_sha=digest(path)
    rows=source_rows(parse_source(path.read_bytes(),osu_sha).objects)
    if rows!=result.rows:raise RuntimeError('Export/reparse changed committed rows')
    duration=case['asset']['duration_ms'];trace=ChartTrace(rows,duration+1)
    objects=parse_osu_file(path).hit_objects
    scopes=[Scope(s['name'],s['start_ms'],s['end_ms']) for s in case['scopes']]
    report=evaluate_scopes(trace,scopes,mel=np.load(case['asset']['mel_file']),identity=dict(identity,
        case=case['key'],seed=case['seed'],osu_sha256=osu_sha,audio_sha256=case['asset']['audio_sha256'],
        mel_sha256=case['asset']['mel_sha256'],controls=result.metrics['controls'],
        head_source=result.metrics['head_source'],completed=True))
    checks=[];contexts=[dict(c,origin='declared',review_status='unreviewed') for c in case.get('review_contexts',())]
    def check(name,passed,observed,limit):
        checks.append(dict(name=name,passed=bool(passed),observed=observed,limit=limit))
    short_all=[dict(column=column,previous_ms=float(times[i]),attack_ms=float(times[i+1]),
                    gap_ms=float(times[i+1]-times[i])) for column,times in enumerate(trace.heads)
               for i in np.flatnonzero(np.diff(times)<20)]
    check('below20ms_attacks_all_coverage',not short_all,len(short_all),0)
    report['short_attacks_all_coverage']=short_all
    for target,measurement,scope in zip(case['scopes'],report['scopes'],scopes):
        stars=target.get('stars');rho=target.get('ln_fraction')
        measurement['requested_stars']=stars;measurement['requested_LN_fraction']=rho
        measurement['difficulty_proxy']=asdict(scoped_difficulty(objects,scope.start_ms,scope.end_ms))
        check(scope.name+':below20ms_attacks',not measurement['short_attacks'],len(measurement['short_attacks']),0)
        if stars is not None:
            error=abs(measurement['difficulty_proxy']['level']-stars)
            check(scope.name+':difficulty',error<=settings.difficulty_error_limit,error,settings.difficulty_error_limit)
        if rho is not None:
            observed=measurement['LN_head_fraction'];error=None if observed is None else abs(observed-rho)
            check(scope.name+':LN_amount',error is not None and error<=settings.ln_fraction_error_limit,error,
                  settings.ln_fraction_error_limit)
        if envelope is not None:
            witness=sustained_attack_witnesses(trace,scope,envelope,stars)
            measurement['pressure_witnesses']=witness
            contexts.extend(pressure_review_contexts(witness,trace.coverage_ms))
            if target.get('maximum_excess_seconds') is not None:
                limit=target['maximum_excess_seconds']
                value=witness.get('excess_seconds')
                check(scope.name+':pressure_excess',value is not None and value<=limit,value,limit)
    metrics=result.metrics;startup=metrics['operational_startup_seconds']
    check('startup',startup is not None and startup<=settings.startup_seconds_limit,startup,settings.startup_seconds_limit)
    check('service',metrics['maximum_service_seconds']<=settings.service_seconds_limit,
          metrics['maximum_service_seconds'],settings.service_seconds_limit)
    check('publication_deadlines',deadlines.get('trace_meets_deadlines',False),deadlines.get('deadline_misses'),0)
    report['publication']=deadlines;report['checks']=checks;report['review_contexts']=contexts
    brief=[{k:v for k,v in scope.items() if k!='contrasts'} for scope in report['scopes']]
    record=dict(key=case['key'],seed=case['seed'],completed=True,osu_file=str(path),osu_sha256=osu_sha,
        whole_stars=None if case.get('switch') is not None else compute_mania_star_rating_20241007(objects,4,1),
        scopes=brief,publication=deadlines,first30_seconds=metrics['first30_rows_seconds'],
        operational_startup_seconds=startup,maximum_service_seconds=metrics['maximum_service_seconds'],
        generation_seconds=metrics['generation_seconds'],audio_seconds=metrics['audio_encode_seconds'],
        head_source=metrics['head_source'],checks=checks,numeric_status='passed' if all(c['passed'] for c in checks) else 'failed',
        semantic_review='pending',review_contexts=contexts,identity=report['identity'])
    return report,record


def run_qualification(settings,*,resolved_yaml=None,on_case=None):
    """Run complete candidate cases into a fresh directory without promoting it.

    Numeric failure retains exports and returns candidate_status='failed'. A
    numerically clear result returns 'review_required', never 'qualified'. Runtime
    failures retain completed earlier cases and a failed-case record. Corpus-
    envelope excess is diagnostic unless a scope declares an explicit bound.
    """
    settings.validate();_verified(settings.checkpoint_file,settings.checkpoint_sha256)
    _verified(settings.plan_file,settings.plan_sha256);plan=_plan(settings.plan_file)
    envelope=None
    if settings.envelope_file is not None:
        _verified(settings.envelope_file,settings.envelope_sha256)
        envelope=AttackEnvelope(**json.loads(Path(settings.envelope_file).read_text())['envelope'])
    if any(s.get('maximum_excess_seconds') is not None and (envelope is None or s.get('stars') is None)
           for case in plan['cases'] for s in case['scopes']):
        raise ValueError('A pressure bound requires an envelope and an explicit scope difficulty')
    out=Path(settings.output_dir);out.mkdir(parents=True,exist_ok=False)
    _write(out/'settings.json',asdict(settings));_write(out/'plan.json',plan)
    if resolved_yaml is not None:(out/'resolved.yaml').write_text(resolved_yaml)
    torch.set_num_threads(settings.cpu_threads)
    payload=torch.load(settings.checkpoint_file,map_location='cpu',weights_only=True);family=payload.get('format');del payload
    if family=='controlled-audio-memory/v1':model=load_memory(settings.checkpoint_file);session_type=MemorySession
    elif family=='controlled-audio/v1':model=load_model(settings.checkpoint_file);session_type=ControlledSession
    else:raise ValueError('Qualification supports controlled-audio/v1 and controlled-audio-memory/v1')
    if plan.get('recovery') is not None:
        recovery=Recovery(**plan['recovery'])
        if recovery.hh!=model.config.minimum_action_gap_ms:
            raise ValueError('Recovery HH must match the checkpoint H-capacity law')
        model.recovery=recovery
    try:
        revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
        source_status=subprocess.check_output(['git','status','--porcelain','--','src','tests','pyproject.toml','uv.lock'],
                                              text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):revision=source_status=None
    identity=dict(checkpoint_sha256=settings.checkpoint_sha256,checkpoint_family=family,plan_sha256=settings.plan_sha256,
        evaluator_source_revision=revision,executable_source_status=source_status,
        recovery=asdict(model.recovery),ln_feedback=settings.ln_feedback,
        runtime=dict(torch=str(torch.__version__),device='cpu',threads=settings.cpu_threads,
                     memory_guard='Darwin task footprint; RSS on other platforms'))
    _write(out/'identity.json',identity)
    records=[];started=time.perf_counter();last_memory_check=started
    def guard():
        nonlocal last_memory_check
        now=time.perf_counter()
        if now-started>settings.max_seconds or (out/'STOP').exists():raise _BudgetStop('qualification_time_or_explicit_stop')
        if now-last_memory_check>=1:
            last_memory_check=now
            observed=footprint_bytes()
            if observed is None:observed=psutil.Process().memory_info().rss
            if observed>settings.footprint_limit_bytes:raise _BudgetStop('qualification_memory_limit')
    for case in plan['cases']:
        try:
            guard();result,deadlines,publications=_generate(model,session_type,case,settings,guard)
            saved=save_rollout(out/case['key'],result,audio_file=case['asset']['audio_file'])
            _write(out/case['key']/'publications.json',publications)
            if not result.completed:
                record=dict(key=case['key'],completed=False,reason=result.stop_reason,coverage_ms=result.coverage_ms,
                    publication=deadlines,numeric_status='failed',semantic_review='unreviewed')
            else:
                report,record=_inspect(case,saved,result,deadlines,settings,envelope,identity)
                _write(out/case['key']/'evaluation.json',report)
        except Exception as error:
            record=dict(key=case['key'],completed=False,reason=type(error).__name__+': '+str(error),
                        numeric_status='failed',semantic_review='unreviewed')
        records.append(record);_write(out/'cases.json',records)
        if on_case is not None:on_case(record)
        if not record['completed']:break
    complete=len(records)==len(plan['cases']) and all(r['completed'] for r in records)
    numeric=complete and all(r['numeric_status']=='passed' for r in records)
    summary=dict(format='native-gameplay-qualification-result/v1',run_status='complete' if complete else 'incomplete',
        candidate_status='review_required' if numeric else 'failed',promoted=False,
        semantic_review='pending' if complete else 'unreviewed',cases_completed=sum(r['completed'] for r in records),
        cases_planned=len(plan['cases']),seconds=time.perf_counter()-started,
        cases_sha256=digest(out/'cases.json'),identity=identity,
        failed_cases=[r['key'] for r in records if r['numeric_status']=='failed'],
        unrun_cases=[c['key'] for c in plan['cases'][len(records):]])
    _write(out/'result.json',summary);return summary
