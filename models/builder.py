import os
from functools import partial
from xml.parsers.expat import model
import timm
from .timm_wrapper import TimmCNNEncoder
import torch
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms
import torch.nn as nn
import torchvision

def has_CONCH():
    HAS_CONCH = False
    CONCH_CKPT_PATH = ''
    # check if CONCH_CKPT_PATH is set and conch is installed, catch exception if not
    try:
        from conch.open_clip_custom import create_model_from_pretrained
        # check if CONCH_CKPT_PATH is set
        if 'CONCH_CKPT_PATH' not in os.environ:
            raise ValueError('CONCH_CKPT_PATH not set')
        HAS_CONCH = True
        CONCH_CKPT_PATH = os.environ['CONCH_CKPT_PATH']
    except Exception as e:
        print(e)
        print('CONCH not installed or CONCH_CKPT_PATH not set')
    return HAS_CONCH, CONCH_CKPT_PATH

def has_UNI():
    HAS_UNI = False
    UNI_CKPT_PATH = ''
    # check if UNI_CKPT_PATH is set, catch exception if not
    try:
        # check if UNI_CKPT_PATH is set
        if 'UNI_CKPT_PATH' not in os.environ:
            raise ValueError('UNI_CKPT_PATH not set')
        HAS_UNI = True
        UNI_CKPT_PATH = os.environ['UNI_CKPT_PATH']
    except Exception as e:
        print(e)
    return HAS_UNI, UNI_CKPT_PATH
        
def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    if model_name == 'resnet50_trunc':
        model = TimmCNNEncoder()
    elif model_name == 'uni_v1':
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        assert HAS_UNI, 'UNI is not available'
        model = timm.create_model("vit_large_patch16_224",
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)
    elif model_name == 'conch_v1':
        HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
        assert HAS_CONCH, 'CONCH is not available'
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    ##addition here
    elif model_name == 'Hibou-B':
        ##option 1
        # from transformers import AutoImageProcessor, AutoModel

        # processor = AutoImageProcessor.from_pretrained("histai/hibou-b", trust_remote_code=True)
        # model = AutoModel.from_pretrained("histai/hibou-b", trust_remote_code=True)
        
        # #model.forward = partial(model.forward, return_dict=True)

        ##option 2
        from hibou import build_model

        hibou_path='./hibou/hibou-b.pth'

        model = build_model(weights_path=hibou_path)

        processor= torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.7068, 0.5755, 0.7220], std=[0.1950, 0.2316, 0.1816]),
        ])

        return model, processor
    
    elif model_name=='PathoDuet':
        from .vits import VisionTransformerMoCo

        processor= torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop((224, 224)),
            torchvision.transforms.ToTensor(),
        ])


        # init the model
        model = VisionTransformerMoCo(pretext_token=True, global_pool='avg')
        model.head= nn.Identity()
        pathoduet_ckpt_path = "./pathoduet/checkpoint_HE.pth"
        checkpoint = torch.load(pathoduet_ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
        # # init the fc layer
        # model.head = nn.Linear(768, args.num_classes)
     
        
        return model, processor
    
    elif model_name=='phikon_v2':
        from transformers import AutoImageProcessor, AutoModel

        MODEL_DIR = "./phikon-v2"

        processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
        model = AutoModel.from_pretrained(MODEL_DIR)

        return model, processor

       

    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    
    constants = MODEL2CONSTANTS[model_name]
    img_transforms = get_eval_transforms(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size = target_img_size)
    if model_name == 'Hibou-B':
        return model, processor
    
    return model, img_transforms