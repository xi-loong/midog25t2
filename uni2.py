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


class VPTViT(nn.Module):
    """
    Wrap a timm VisionTransformer with Visual Prompt Tuning (VPT).
    - Supports shallow (only before block 0) and deep (before every block).
    - Freeze the backbone; train only prompts (+ optional head).
    """

    def __init__(
            self,
            vit,
            num_classes=None,
            prompt_len: int = 8,
            deep: bool = True,
            prompt_dropout: float = 0.0,
            train_head: bool = True,
    ):
        super().__init__()
        assert isinstance(vit, timm.models.VisionTransformer)
        self.vit = vit
        self.embed_dim = vit.embed_dim
        self.prompt_len = prompt_len
        self.deep = deep
        self.prompt_dropout = nn.Dropout(prompt_dropout) if prompt_dropout > 0 else nn.Identity()

        # 可选：重置分类头类别数
        if num_classes is not None and num_classes != vit.num_classes:
            vit.reset_classifier(num_classes=num_classes)

        # === 构造提示参数 ===
        if deep:
            # 每一层一个独立的 prompt 参数
            self.deep_prompts = nn.ParameterList([
                nn.Parameter(torch.zeros(1, prompt_len, self.embed_dim))
                for _ in range(len(vit.blocks))
            ])
            nn.init.trunc_normal_(self.deep_prompts[0], std=0.02)
            for p in self.deep_prompts[1:]:
                nn.init.trunc_normal_(p, std=0.02)
        else:
            # 只在第0层前插一次
            self.shallow_prompt = nn.Parameter(torch.zeros(1, prompt_len, self.embed_dim))
            nn.init.trunc_normal_(self.shallow_prompt, std=0.02)

        # === 冻结骨干参数，仅训练 prompt（和可选 head）===
        for n, p in self.vit.named_parameters():
            p.requires_grad = False
        if train_head:
            # 允许 head 更新（更稳）
            for n, p in self.vit.get_classifier().named_parameters():
                p.requires_grad = True

    @torch.no_grad()
    def _freeze_backbone_buffers(self):
        # 运行时不更新 pos_embed/cls/reg 等缓冲
        self.vit.eval()

    def _prep_inputs(self, x: torch.Tensor):
        """
        复制自 vit.forward_features 的前半段：得到加入 pos_embed 的序列 tokens。
        """
        x = self.vit.patch_embed(x)  # [B, N_patches, C] 或 NHWC -> NLC
        x = self.vit._pos_embed(x)  # 加上 pos_embed 并拼接 cls/reg
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)
        return x

    def _insert_prompts(self, x: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        """
        在 prefix tokens（cls+reg）之后插入 prompt，再接原 patch tokens。
        x: [B, N_prefix + N_patch, C]
        prompt: [1, P, C]
        """
        B = x.size(0)
        P = prompt.size(1)
        num_prefix = self.vit.num_prefix_tokens  # cls + reg (可能为1或>1)
        prefix = x[:, :num_prefix, :]
        patches = x[:, num_prefix:, :]
        prompt_b = prompt.expand(B, P, -1)
        x = torch.cat([prefix, self.prompt_dropout(prompt_b), patches], dim=1)
        return x

    def _remove_prompts(self, x: torch.Tensor, P: int) -> torch.Tensor:
        """
        从序列中移除刚才插入的 P 个 prompt。
        """
        num_prefix = self.vit.num_prefix_tokens
        prefix = x[:, :num_prefix, :]
        # 移除 [num_prefix : num_prefix+P)
        patches = x[:, num_prefix + P:, :]
        return torch.cat([prefix, patches], dim=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        复现 vit.forward_features，但在每一层前插入/移除 prompt（VPT）。
        """
        x = self._prep_inputs(x)

        if self.deep:
            # 层前插入 -> 过 block -> 移除
            for i, blk in enumerate(self.vit.blocks):
                x = self._insert_prompts(x, self.deep_prompts[i])
                x = blk(x)  # 注意：若你使用 attn_mask/grad ckpt，按需改为与原实现一致
                x = self._remove_prompts(x, self.prompt_len)
        else:
            # 只在第0层前插入一次
            x = self._insert_prompts(x, self.shallow_prompt)
            for i, blk in enumerate(self.vit.blocks):
                x = blk(x)
            x = self._remove_prompts(x, self.prompt_len)

        x = self.vit.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.vit.forward_head(x)  # 池化+fc_norm+dropout+head
        return x


def build(num_classes=2, fine_tuning=dict(type='lora',alpha=16, rank=32)):
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

    if fine_tuning:
        type = fine_tuning.pop('type')
        if type == 'lora':
            peft_config = LoraConfig(
                r=fine_tuning['rank'],
                lora_alpha=fine_tuning['alpha'],
                lora_dropout=0.1,
                target_modules=["qkv", "proj", "mlp.fc1", "mlp.fc2"],
                modules_to_save=[]
            )
            backbone = get_peft_model(backbone, peft_config)
        elif type == 'vpt':
            backbone = VPTViT(backbone, **fine_tuning)
        model = ViTClassifier(backbone, 1536, num_classes, frozen=False)
    else:
        model = ViTClassifier(backbone, 1536, num_classes, frozen=True)

    return model


def load(checkpoint_path, num_classes=2, fine_tuning=dict(type='lora', alpha=16, rank=32), device="cpu"):
    # Step 1: 构建模型（本地，不加载预训练）
    model = build(num_classes=num_classes, fine_tuning=fine_tuning)

    # Step 2: 加载 safetensors 权重
    state_dict = load_file(checkpoint_path, device=device)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()
    return model


pipeline = A.Compose([
    A.CenterCrop(width=128, height=128, pad_if_needed=True, fill=255),
    A.Resize(width=224, height=224),
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
            if 'midog25' in path:
                path = path.replace('midog25', 'midog25-stain')
            elif 'AtypicalMitoses' in path:
                path = path.replace('AtypicalMitoses', 'AtypicalMitoses-stain')
            elif 'AmiBr' in path:
                path = path.replace('AmiBr', 'AmiBr-stain')
            else:
                raise NotImplementedError
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image, cls

        def __len__(self):
            return len(self.paths)


    model = load(
        r'E:\work\mitosis\uni2-mitosis-classifier-vpt-stain-norm0.5-resize0.9_1.1-prompt8\checkpoint-14807\model.safetensors',
        fine_tuning=dict(type='vpt'),
    )
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
