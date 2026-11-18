"""Portable, inference-only adapter for the selected R19/R29/R18 combination.

The model definitions and input recipe are copied unchanged in project/common.
Only weight path resolution is portable here. No training or network callbacks.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'project'))
from common.portable_hand_evidence import PortableHandEvidence, sha256_file

EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
         (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
         (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]

def main():
    parser = argparse.ArgumentParser(description='R19 primary + R29 no-hand fallback + R18 hand21, inference only')
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', required=True, help='new result directory')
    parser.add_argument('--models', default=str(HERE / 'models'))
    parser.add_argument('--roi', help='JSON [x,y,width,height], normalized; native 4K default uses verified camera ROI')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('output already exists; choose a new result directory')
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    image = Image.open(args.image).convert('RGB')
    roi = json.loads(args.roi) if args.roi else ([.375, .1, .375, 1400/3000] if image.size == (4000, 3000) else None)
    models = Path(args.models)
    inventory = json.loads((models / 'model_inventory.json').read_text(encoding='utf-8-sig'))
    fallback = next(x for x in inventory['files'] if x['role'] == 'fallback_detector')
    model = PortableHandEvidence(models / 'glove_evidence/manifest.json', models / fallback['file'],
                                 fallback['sha256'], device=args.device, roi_normalized=roi)
    bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
    result = model.predict(bgr)
    result.update(input_sha256=sha256_file(args.image), input_size=list(image.size), camera_roi=roi,
                  optimizer_steps=0, z_not_estimated=True, joint_visibility_not_estimated=True,
                  hand_gesture_class_not_estimated=True, actual_connection_not_estimated=True)
    output.mkdir(parents=True)
    (output / 'predictions.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    draw = ImageDraw.Draw(image)
    for i, hand in enumerate(result['hands']):
        color = [(0, 200, 170), (255, 170, 50), (100, 180, 255)][i % 3]
        pts = hand['xy_source']
        draw.rectangle(hand['box_xyxy'], outline=color, width=1)
        for a, b in EDGES:
            draw.line([tuple(pts[a]), tuple(pts[b])], fill=color, width=1)
        for x, y in pts:
            draw.ellipse((x-1.2, y-1.2, x+1.2, y+1.2), fill=color)
    image.save(output / 'prediction.png')
    print(json.dumps(dict(output=str(output.resolve()), hands=len(result['hands']),
                          fallback_used=result['R29_fallback_used'], new_optimizer_steps=0), ensure_ascii=False))

if __name__ == '__main__':
    main()
