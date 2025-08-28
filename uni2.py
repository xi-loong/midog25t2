import json
import glob

import numpy as np
import cv2

import torch
import tqdm
from peft import LoraConfig, get_peft_model
from torch import nn
from timm.models.vision_transformer import VisionTransformer
import timm.layers
from safetensors.torch import load_file

import albumentations as A


class ViTClassifier(torch.nn.Module):
    def __init__(self, backbone, in_channels, out_channels=2, frozen=False):
        super().__init__()
        self.backbone = backbone
        self.frozen = frozen
        self.classifier = nn.Linear(in_channels, out_channels)

        # 如果整体 frozen=True（不使用 LoRA），把 backbone 的参数 freeze（但仍应允许 train/eval 切换）
        if self.frozen:
            for n, p in self.backbone.named_parameters():
                p.requires_grad = False

    def forward(self, pixel_values=None, labels=None, **kwargs):
        # 接受关键字参数以兼容 transformers.Trainer 的输入 dict
        # backbone 应返回 (B, 1536) tensor. 如果返回结构不同，请 adapt。
        if self.frozen:
            with torch.no_grad():
                features = self.backbone(pixel_values)
        else:
            features = self.backbone(pixel_values)

        logits = self.classifier(features)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {'loss': loss, 'logits': logits}


def build(num_classes=2, lora_cfg=dict(alpha=16, rank=32)):
    backbone = VisionTransformer(
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        embed_dim=1536,
        init_values=1e-5,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )
    if lora_cfg:
        peft_config = LoraConfig(
            r=lora_cfg['rank'],
            lora_alpha=lora_cfg['alpha'],
            lora_dropout=0.1,
            target_modules=["qkv", "proj", "mlp.fc1", "mlp.fc2"],
            modules_to_save=[]
        )
        backbone = get_peft_model(backbone, peft_config)
        model = ViTClassifier(backbone, 1536, num_classes, frozen=False)
    else:
        model = ViTClassifier(backbone, 1536, num_classes, frozen=True)

    return model


def load(checkpoint_path, num_classes=2, lora_cfg=dict(alpha=16, rank=32), device="cpu"):
    # Step 1: 构建模型（本地，不加载预训练）
    model = build(num_classes=num_classes, lora_cfg=lora_cfg)

    # Step 2: 加载 safetensors 权重
    state_dict = load_file(checkpoint_path, device=device)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()
    return model


pipeline = A.Compose([
    A.Resize(width=224, height=224),
    A.CenterCrop(width=224, height=224),
    A.Normalize(mean=(0.4850, 0.4560, 0.4060), std=(0.2290, 0.2240, 0.2250))
])


def preprocess(image, use_tta=True):
    ouputs = []
    if use_tta:
        rotates = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
        for img in [image, cv2.flip(image, 0)]:
            for angle in rotates:
                tta = img if angle is None else cv2.rotate(img, angle)
                norm = pipeline(image=tta)['image']
                norm = np.transpose(norm, axes=[2, 0, 1])
                ouputs.append(norm)
    else:
        norm = pipeline(image=image)['image']
        norm = np.transpose(norm, axes=[2, 0, 1])
        ouputs.append(norm)
    return np.stack(ouputs, axis=0)


if __name__ == '__main__':
    from sklearn.metrics import balanced_accuracy_score
    from torch.utils import data


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return balanced_accuracy_score(labels, predictions)


    class Dataset(data.Dataset):
        def __init__(self, path):
            super().__init__()
            self.paths = []
            for p in glob.glob(path):
                self.paths += json.load(open(p))

        def __getitem__(self, item):
            cls, path = self.paths[item]
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image, cls

        def __len__(self):
            return len(self.paths)


    model = load(r'E:\work\mitosis\uni2-mitosis-classifier-lora2\checkpoint-2613\model.safetensors')
    dataset = Dataset('group1.json')
    model = model.cuda()
    labels = []
    outputs = []
    for data, label in tqdm.tqdm(dataset):
        inputs = torch.from_numpy(preprocess(data, True)).cuda()
        with torch.inference_mode():
            pred = model(inputs)['logits'].mean(dim=0).cpu().numpy()
            outputs.append(pred)
        labels.append(label)
    outputs = np.stack(outputs, axis=0)
    print(compute_metrics((outputs, labels)))
