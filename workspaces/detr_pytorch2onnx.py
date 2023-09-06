import torch
import onnx
import onnxsim

from mmdet.apis import init_detector
from mmdet.core import bbox_cxcywh_to_xyxy

import torch.nn.functional as F

class DetrONNX(torch.nn.Module):
    def __init__(self, model):
        super(DetrONNX, self).__init__()
        self.backbone = model.backbone
        self.head = model.bbox_head
        self.max_per_img = 100

    def forward(self, img):
        x = self.backbone(img)[0]
        x = self.head.input_proj(x)

        h, w = x.size()[-2:]
        masks = x.new_zeros((1, h, w))
        masks = F.interpolate(masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
        pos_embed = self.head.positional_encoding(masks)
        outs_dec, _ = self.head.transformer(x, masks, self.head.query_embedding.weight, pos_embed)  # (1, 100, 256)

        all_cls_scores = self.head.fc_cls(outs_dec)
        all_bbox_preds = self.head.fc_reg(self.head.activate(self.head.reg_ffn(outs_dec))).sigmoid()
        cls_score = all_cls_scores[5][0]    # (100, 81)
        bbox_pred = all_bbox_preds[5][0]    # (100, 4)

        scores, det_labels = torch.nn.functional.softmax(cls_score, dim=-1)[..., :-1].max(-1)
        scores, bbox_index = scores.topk(self.max_per_img)
        bbox_pred = bbox_pred[bbox_index]
        det_labels = det_labels[bbox_index]
        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(-1)), -1)  # (100, 5)

        return det_bboxes, det_labels.float()

    def export_onnx(self, onnx_path):
        dummy_img = torch.randn(1, 3, 800, 1199, device='cuda:0')
        torch.onnx.export(self, dummy_img, onnx_path, opset_version=11)
        print('Saved DETR to onnx file: {}'.format(onnx_path))

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        onnx_sim, _ = onnxsim.simplify(onnx_model)
        onnx.save(onnx_sim, onnx_path)


if __name__ == '__main__':
    # https://github.com/open-mmlab/mmdetection/tree/master/configs/detr
    config_file = '../configs/detr/detr_r50_8x2_150e_coco.py'
    checkpoint_file = '../checkpoints/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'
    checkpoint_model = init_detector(config_file, checkpoint_file)

    detr_model = DetrONNX(checkpoint_model)
    detr_model.export_onnx('./detr.onnx')