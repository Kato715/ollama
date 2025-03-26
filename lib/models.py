import torchvision
import torch.nn as nn
import torch
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from lib.utils import Dict
import os

def load_model(conf: Dict):
    model = select(conf.model.type, conf.model.n_classes)
    model_path = os.path.join(conf.model.dir,
                      f"{conf.model.id}_{conf.model.type}_{conf.model.size[0]}_{conf.model.size[1]}_{conf.datasets.train.val_fold}.bin")

    # model.load_state_dict(torch.load(
    #    model_path,
    #    map_location=lambda storage, loc: storage.cuda() if torch.cuda.is_available() else 'cpu')
    # )
    model.load_state_dict(torch.load(
       model_path,
       map_location=torch.device('cpu')       
    )
    )
    model.to(conf.model.device)

    return model

def select(model_type: str, num_classes: int):
    if model_type == "resnet18":
        return Resnet18(num_classes)
    elif model_type == "resnet50":
        return Resnet50(num_classes)
    elif model_type == "resnet152":
        return Resnet152(num_classes)
    elif model_type == "vgg11":
        return VGG11(num_classes)        
    elif model_type == "efficientnetb0":
        return EfficientNetB0(num_classes)
    elif model_type == "efficientnetb7":
        return EfficientNetB7(num_classes)
    elif model_type == "convnext-tiny":
        return ConvnextTiny(num_classes)
    elif model_type == "convnext-small":
        return ConvnextSmall(num_classes)
    elif model_type == "convnext-base":
        return ConvnextBase(num_classes)
    elif model_type == "convnext-large":
        return ConvnextLarge(num_classes)
    elif model_type == "vitb16-ours":
        return VitOurs(num_classes)
    elif model_type == "vitb16":
        return VitB16(num_classes)
    elif model_type == "vitb32":
        return VitB32(num_classes)
    elif model_type == "vitl16":
        return VitL16(num_classes)
    elif model_type == "vitl32":
        return VitL32(num_classes)
    elif model_type == "efficientnetb0":
        return EfficientNetB0(num_classes)
    elif model_type == "efficientnetb4":
        return EfficientNetB4(num_classes)
    elif model_type == "efficientnetb7":
        return EfficientNetB7(num_classes)
    elif model_type == "densenet121":
        return DenseNet121(num_classes)
    elif model_type == "densenet169":
        return DenseNet169(num_classes)
    
    else:
        raise ValueError(f"Unsupported model type.")


class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super(Resnet18, self).__init__()
        self.convnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.convnet.fc = nn.Linear(512, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        return loss

    def forward(self, image, targets=None):
        outputs = self.convnet(image)
        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss
        return outputs, None


class Resnet50(nn.Module):
    def __init__(self, num_classes):
        super(Resnet50, self).__init__()
        self.convnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.convnet.fc = nn.Linear(2048, num_classes) 

    def loss(self, outputs, targets):
        if targets is None:
            return None
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        return loss

    def forward(self, image, targets=None):
        outputs = self.convnet(image)
        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss
        return outputs, None


class Resnet152(nn.Module):
    def __init__(self, num_classes):
        super(Resnet152, self).__init__()
        self.convnet = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1)
        self.convnet.fc = nn.Linear(2048, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        return loss

    def forward(self, image, targets=None):
        outputs = self.convnet(image)
        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss
        return outputs, None


class VGG11(nn.Module):
    def __init__(self, num_classes):
        super(VGG11, self).__init__()
        self.base = torchvision.models.vgg11(weights=torchvision.models.VGG11_Weights.IMAGENET1K_V1)
        self.base.classifier[6] = nn.Linear(4096, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        return loss

    def forward(self, image, targets=None):
        outputs = self.base(image)
        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss
        return outputs, None


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB0, self).__init__()
        self.base = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.base.classifier[1] = nn.Linear(1280, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        return loss

    def forward(self, image, targets=None):
        outputs = self.base(image)
        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss
        return outputs, None


class EfficientNetB7(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB7, self).__init__()
        self.base = torchvision.models.efficientnet_b7(weights=torchvision.models.EfficientNet_B7_Weights.IMAGENET1K_V1)
        self.base.classifier[1] = nn.Linear(2560, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        return loss

    def forward(self, image, targets=None):
        outputs = self.base(image)
        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss
        return outputs, None


class ConvnextTiny(nn.Module):
    def __init__(self, num_classes):
        super(ConvnextTiny, self).__init__()
        self.base = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.base.classifier[2] = nn.Linear(768, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        return loss

    def forward(self, image, targets=None):
        outputs = self.base(image)
        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss
        return outputs, None


class ConvnextSmall(nn.Module):
    def __init__(self, num_classes):
        super(ConvnextSmall, self).__init__()
        self.base = torchvision.models.convnext_small(weights=torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
        self.base.classifier[2] = nn.Linear(768, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        return loss

    def forward(self, image, targets=None):
        outputs = self.base(image)
        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss
        return outputs, None


class ConvnextBase(nn.Module):
    def __init__(self, num_classes):
        super(ConvnextBase, self).__init__()
        self.base = torchvision.models.convnext_base(weights=torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.base.classifier[2] = nn.Linear(1024, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        return loss

    def forward(self, image, targets=None):
        outputs = self.base(image)
        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss
        return outputs, None


class ConvnextLarge(nn.Module):
    def __init__(self, num_classes):
        super(ConvnextLarge, self).__init__()
        self.base = torchvision.models.convnext_large(weights=torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
        self.base.classifier[2] = nn.Linear(1536, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        return loss

    def forward(self, image, targets=None):
        outputs = self.base(image)
        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss
        return outputs, None

class VitOurs(nn.Module):
    
    def __init__(self, num_classes, pretrained_path=None):
        """
        ViT_Ours モデルを初期化し、AGCAM を統合するクラス
        
        Args:
            model_name (str): モデルの名前 (例: 'vit_base_patch16_224')
            num_classes (int): 出力クラス数
            pretrained_path (str, optional): 事前学習済みモデルのパス
        """
        super(VitOurs, self).__init__()
        
        # ViT_Ours モデルのロード
        self.base = ViT_Ours.create_model(vit_base_patch16_224, pretrained=True, num_classes=num_classes)
        
        # 事前学習済みの重みをロード（必要な場合）
        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            self.base.load_state_dict(state_dict, strict=True)
        
        # モデルを評価モードに設定
        self.base.eval()
        
        # AGCAM の初期化
        self.agcam = AGCAM(self.base)
    
    def loss(self, outputs, targets):
        """
        損失計算用の関数
        """
        if targets is None:
            return None
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        return loss
    
    def forward(self, image, targets=None):
        """
        フォワードパス
        """
        outputs = self.base(image)
        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss
        return outputs, None

class VitB16(nn.Module):
    def __init__(self, num_classes):
        super(VitB16, self).__init__()
        self.base = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.base.heads = nn.Linear(768, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        return loss

    def forward(self, image, targets=None):
        outputs = self.base(image)
        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss
        return outputs, None


class VitB32(nn.Module):
    def __init__(self, num_classes):
        super(VitB32, self).__init__()
        self.base = torchvision.models.vit_b_32(weights=torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1)
        self.base.heads = nn.Linear(768, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        return loss

    def forward(self, image, targets=None):
        outputs = self.base(image)
        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss
        return outputs, None

class VitL16(nn.Module):
    def __init__(self, num_classes):
        super(VitL16, self).__init__()
        self.base = torchvision.models.vit_l_16(weights=torchvision.models.ViT_L_16_Weights.IMAGENET1K_V1)
        self.base.heads = nn.Linear(1024, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        return loss

    def forward(self, image, targets=None):
        outputs = self.base(image)
        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss
        return outputs, None

class VitL32(nn.Module):
    def __init__(self, num_classes):
        super(VitL32, self).__init__()
        self.base = torchvision.models.vit_l_32(weights=torchvision.models.ViT_L_32_Weights.IMAGENET1K_V1)
        self.base.heads = nn.Linear(1024, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        return loss

    def forward(self, image, targets=None):
        outputs = self.base(image)
        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss
        return outputs, None

# EfficientNetB4クラス
class EfficientNetB4(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB4, self).__init__()
        self.base = torchvision.models.efficientnet_b4(weights=torchvision.models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        self.base.classifier[1] = nn.Linear(1792, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        return loss

    def forward(self, image, targets=None):
        outputs = self.base(image)
        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss
        return outputs, None

# DenseNet121クラス
class DenseNet121(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet121, self).__init__()
        self.base = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
        self.base.classifier = nn.Linear(1024, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        return loss

    def forward(self, image, targets=None):
        outputs = self.base(image)
        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss
        return outputs, None

# DenseNet169クラス
class DenseNet169(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet169, self).__init__()
        self.base = torchvision.models.densenet169(weights=torchvision.models.DenseNet169_Weights.IMAGENET1K_V1)
        self.base.classifier = nn.Linear(1664, num_classes)

    def loss(self, outputs, targets):
        if targets is None:
            return None
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        return loss

    def forward(self, image, targets=None):
        outputs = self.base(image)
        if targets is not None:
            loss = self.loss(outputs, targets)
            return outputs, loss
        return outputs, None