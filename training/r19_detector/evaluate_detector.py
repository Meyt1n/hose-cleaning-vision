# -*- coding: utf-8 -*-
import torch,collections,time
from torchvision.ops import box_iou


@torch.inference_mode()
def evaluate(model,data,cfg):
    model.eval();cases=[];groups=collections.defaultdict(lambda:{'images':0,'GT_boxes':0,'matched_at_50':0,'matched_at_75':0,'extra_unmatched':0,'predictions':0})
    started=time.monotonic()
    for r in data.evaluation_rows():
        image,target=data.load(r);prediction=model([image.cuda()])[0]
        keep=(prediction['scores']>=cfg['score_threshold'])&(prediction['labels']==1)
        boxes=prediction['boxes'][keep].cpu();scores=prediction['scores'][keep].cpu()
        gt=target['boxes'];ious=box_iou(boxes,gt)
        matched={}
        for threshold in [.5,.75]:
            used=set();pairs=[]
            for j in scores.argsort(descending=True).tolist():
                available=[g for g in range(len(gt)) if g not in used]
                if not available:break
                best=max(available,key=lambda g:float(ious[j,g]))
                if ious[j,best]>=threshold:used.add(best);pairs.append([j,best,float(ious[j,best])])
            matched[str(threshold)]=pairs
        n50=len(matched['0.5']);n75=len(matched['0.75'])
        d=groups[r['split']];d['images']+=1;d['GT_boxes']+=len(gt);d['matched_at_50']+=n50;d['matched_at_75']+=n75;d['extra_unmatched']+=len(boxes)-n50;d['predictions']+=len(boxes)
        cases.append({'id':r['id'],'split':r['split'],'boxes_xyxy':boxes.tolist(),'scores':scores.tolist(),'GT_boxes_xyxy':gt.tolist(),
                      'matched':matched,'source_group':r['source_group'],'image_sha256':r['image_sha256']})
    metrics={}
    for split,v in groups.items():
        d=dict(v);d['known_box_recall50']=v['matched_at_50']/max(1,v['GT_boxes']);d['known_box_recall75']=v['matched_at_75']/max(1,v['GT_boxes'])
        if 'field' in split:
            d['unmatched_predictions_are_not_verified_false_positives']=True
            d['precision']=None;d['F1']=None;d['label_scope']='old_AI_coarse_known_boxes;not_independent_acceptance'
        else:
            d['precision']=v['matched_at_50']/max(1,v['predictions']);d['F1']=2*v['matched_at_50']/max(1,v['predictions']+v['GT_boxes'])
            d['label_scope']='author_boxes' if split=='development_hagrid_box' else 'joint_envelope_weak_geometry'
        metrics[split]=d
    return {'metrics':metrics,'cases':cases,'elapsed_seconds':time.monotonic()-started,'workpoint':{'score':cfg['score_threshold'],'NMS':cfg['NMS_IoU'],'max_hands':cfg['max_hands']},'no_independent_field_accuracy':True}
