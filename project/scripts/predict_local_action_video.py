"""Run the actual InferencePipeline on a continuous, timestamped local file.

Example: python scripts/predict_local_action_video.py --config SAFE_PROFILE.json
         --source VIDEO --start 600 --duration 180 --output OUTPUT_DIR
No HTTP server, model training, DB writer or external callback is started.
"""
from __future__ import annotations
from pathlib import Path
from fractions import Fraction
from collections import Counter
import argparse
import hashlib
import json
import sys
import time
import av
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.frame_timing import TimedFrame
from infer_service.config import ServiceConfig
from infer_service.task import Task, StreamSpec, TaskParams
from infer_service.engine.pipeline import InferencePipeline
from infer_service.engine.stream import _AVCapture
from infer_service.engine.annotate import _draw_texts


class RawOnlyPool:
    """A guard: this opt-in backend must never accidentally load the old LSTM."""
    def __init__(self, device): self.device = device
    def get(self, *args, **kwargs): raise RuntimeError('raw backend fell through to old model pool')
    def acquire(self, *args): raise RuntimeError('raw backend acquired old model pool')
    def release(self, *args): raise RuntimeError('raw backend released unacquired old model')


def _atomic(path, value):
    temp=path.with_suffix(path.suffix+'.pending')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def run(config_file, source, start_seconds, duration_seconds, output_dir, progress_callback=None):
    output=Path(output_dir)
    if output.exists():raise ValueError('output exists: reconcile before running again')
    output.mkdir(parents=True)
    config_file=Path(config_file).resolve()
    cfg=ServiceConfig.from_raw(json.loads(config_file.read_text(encoding='utf-8')))
    if not cfg.local_action.enabled:raise ValueError('explicit raw local_action.enabled required')
    source=Path(source).resolve()
    cfg.local_action.manifest=str((config_file.parent / cfg.local_action.manifest).resolve())
    start=Fraction(str(start_seconds));end=start+Fraction(str(duration_seconds))
    warm_start=max(Fraction(0),start-Fraction('2.56'))
    task=Task(id='raw-file-predict',session='local-file',stream=StreamSpec(str(source),protocol='file',fps=25),
              checkpoint=cfg.local_action.manifest,params=TaskParams(infer_fps=12.5,push_fps=25))
    pipeline=InferencePipeline(task,RawOnlyPool(cfg.local_action.device),cfg,
                               pre_rules=cfg.pre_rules,post_rules=cfg.post_rules,
                               station_no='LOCAL_FILE_REVIEW',batch_no='RAW_3MIN_R29_FALLBACK')
    events=[];jpeg_buffer=[]
    pipeline.add_event_sink(lambda x: events.append(dict(x)))
    pipeline.add_frame_sink(jpeg_buffer.append)
    capture=_AVCapture(str(source),True)
    capture.container.streams.video[0].codec_context.thread_count=2
    frames=models=warmup=decoded=0;counts=Counter();first_pts=last_pts=None
    hand_samples=0;elapsed_wall=time.monotonic();latest=None;time_base=None;origin=None
    writer=None;stream=None;last_result_source_pts=None;largest_model_gap=Fraction(0);last_model_time=None
    result_file=output/'predictions.jsonl';display_file=output/'display_alignment.jsonl'
    _atomic(output/'intent.json',dict(source=str(source),range_elapsed_seconds=[float(start),float(end)],
        raw_input=True,original_pipeline=True,optimizer_steps=0))
    def progress(stage):
        x=dict(stage=stage,decoded_native_frames=decoded,model_frames=models,warmup_frames=warmup,
               output_frames=frames,wall_seconds=time.monotonic()-elapsed_wall,optimizer_steps=0,
               source_elapsed_seconds=float((Fraction(last_pts)*time_base-origin) if last_pts is not None else 0))
        _atomic(output/'progress.json',x)
        if progress_callback:progress_callback(x)
    try:
        pipeline.start_offline()
        with result_file.open('x',encoding='utf-8') as trace, display_file.open('x',encoding='utf-8') as align:
            while True:
                native=capture.read_native()
                if native is None:
                    raise ValueError('source ended before requested continuous range')
                decoded+=1;index=decoded-1
                if native.pts is None or native.time_base is None:raise ValueError('source PTS missing')
                tb=Fraction(native.time_base);timestamp=Fraction(native.pts)*tb
                if origin is None:origin=timestamp;time_base=tb
                if tb!=time_base:raise ValueError('source time base changed')
                relative=timestamp-origin
                if relative>=end:break
                last_pts=int(native.pts)
                if relative<warm_start:
                    if decoded%1000==0:progress('SEQUENTIAL_DECODE_TO_RANGE')
                    continue
                scored=relative>=start
                bgr=native.to_ndarray(format='bgr24') if scored or index%2==0 else None
                if index%2==0:
                    packet=TimedFrame(bgr,int(native.pts),str(tb),time.monotonic(),1,index,
                                      native_video_frame=native,score_enabled=scored)
                    latest=pipeline.process_offline(packet)
                    if last_model_time is not None:largest_model_gap=max(largest_model_gap,timestamp-last_model_time)
                    last_model_time=timestamp
                    if scored:
                        models+=1;counts[latest.action_code]+=1
                        hands=latest.hand_evidence.get('hands',[]);hand_samples+=bool(hands)
                        row=dict(source_frame_index=index,source_pts=int(native.pts),time_base=str(tb),
                            source_elapsed_seconds=float(relative),output_seconds=float(relative-start),
                            action_code=latest.action_code,action_name=latest.action_name,
                            operation_probability=latest.model_details['probability'],
                            observation_state=latest.observation_state,decision_reason=latest.decision_reason,
                            causal_history_samples=latest.buffered,
                            candidate_count=pipeline._cleaning_timer.count,
                            current_continuous_action_seconds=float(pipeline._cleaning_timer.duration),
                            hand_candidates=hands,R29_fallback_used=latest.hand_evidence.get('R29_fallback_used',False),elapsed_infer_ms=latest.elapsed_ms,
                            input_origin='current_raw_native_frame',actual_connection_verified=False,per_hose_completion=False)
                        trace.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                        last_result_source_pts=int(native.pts)
                        if models%50==0:trace.flush();progress('RAW_PIPELINE_INFERENCE')
                    else:warmup+=1
                if not scored:continue
                if latest is None:raise ValueError('no causal result at the start of output')
                if latest.source_pts>native.pts:raise ValueError('future prediction displayed on an earlier frame')
                jpeg_buffer.clear()
                pipeline._push_frame(bgr,latest,[])
                if len(jpeg_buffer)!=1:raise ValueError('original pipeline preview renderer failed')
                picture=cv2.imdecode(np.frombuffer(jpeg_buffer[0],np.uint8),cv2.IMREAD_COLOR)
                h,w=picture.shape[:2]
                picture[h-98:]=picture[h-98:]//3
                picture=_draw_texts(picture,[
                    (14,h-94,f'R19主检测+R29补位，动作R26 | 源视频 {float(relative):.2f}s | 本片 {float(relative-start):.2f}s | 清洗作业概率 {latest.model_details["probability"]:.3f}',(255,255,255),23),
                    (14,h-62,f'连续≥3秒动作候选：{pipeline._cleaning_timer.count} | 当前连续证据：{float(pipeline._cleaning_timer.duration):.2f}s | {latest.action_name}',(80,240,255),23),
                    (14,h-30,'动作候选不等于实际清洗根数；未判定每根胶管。原画面旧模型文字不参与本次判断。',(200,200,200),20)])
                if writer is None:
                    writer=av.open(str(output/'prediction.mp4'),mode='w')
                    stream=writer.add_stream('libx264',rate=25)
                    stream.width=w;stream.height=h;stream.pix_fmt='yuv420p'
                    stream.time_base=tb;stream.codec_context.time_base=tb
                    stream.options={'preset':'veryfast','crf':'24','threads':'2'}
                    first_pts=int(native.pts)
                f=av.VideoFrame.from_ndarray(picture,format='bgr24')
                f.pts=int(native.pts)-first_pts;f.time_base=tb
                for p in stream.encode(f):writer.mux(p)
                align.write(json.dumps(dict(output_frame=frames,source_frame_index=index,
                    source_pts=int(native.pts),prediction_source_pts=latest.source_pts,
                    prediction_age_seconds=float(Fraction(int(native.pts)-latest.source_pts)*tb)))+'\n')
                frames+=1
            for p in stream.encode():writer.mux(p)
            writer.close();writer=None
        pipeline.stop('LOCAL_FILE_COMPLETED')
        stats=pipeline._detector.stats()
        assert stats['rgb_forward_calls']==models+warmup==stats['R19_primary_forward_calls']==stats['action_forward_calls']
        assert stats['hand_detector_forward_calls']==models+warmup+stats['R29_fallback_attempts']
        assert stats['cached_feature_inputs']==0
        if not models or not frames or largest_model_gap>Fraction('.12'):raise ValueError('raw output completeness or timing gate failed')
        _atomic(output/'events.json',events)
        result=dict(status='RAW_ORIGINAL_PIPELINE_COMPLETED',source=str(source),
            range_elapsed_seconds=[float(start),float(end)],warmup_frames=warmup,
            output_frames=frames,scored_model_frames=models,decoded_native_frames=decoded,
            first_output_source_pts=first_pts,last_output_source_pts=last_pts,
            source_time_base=str(time_base),max_model_gap_seconds=float(largest_model_gap),
            decisions=dict(counts),accepted_coverage=(models-counts['UNKNOWN'])/models,
            hand_observation_coverage=hand_samples/models,
            research_action_candidate_count=pipeline._cleaning_timer.count,
            detector_stats=stats,wall_seconds=time.monotonic()-elapsed_wall,
            optimizer_steps=0,actual_connection_accuracy=None,per_hose_accuracy=None,
            semantics='cleaning_operation_not_physical_connection',
            no_external_business_transport=True,original_pipeline_infer_loop=True,
            original_pipeline_preview_renderer=True,
            model_sha256=pipeline._loaded.info['checkpoint_sha256'])
        _atomic(output/'completion.json',result);progress('COMPLETED')
        return result
    finally:
        capture.release()
        if writer is not None:writer.close()
        pipeline.stop('LOCAL_FILE_EXIT')


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'config/offline_action.json'))
    a.add_argument('--source',required=True)
    a.add_argument('--start',type=float,default=0);a.add_argument('--duration',type=float,default=180)
    a.add_argument('--output',required=True);args=a.parse_args()
    if not Path(args.source).is_file():
        a.error('--source must be a local video file')
    if not (np.isfinite(args.start) and np.isfinite(args.duration) and args.start >= 0 and args.duration > 0):
        a.error('use a finite nonnegative --start and a finite positive --duration')
    import torch
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    print(json.dumps(run(args.config,args.source,args.start,args.duration,args.output),ensure_ascii=False))
if __name__=='__main__':main()
