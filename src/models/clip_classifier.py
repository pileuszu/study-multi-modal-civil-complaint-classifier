"""
CLIP-based classifier for civil complaints
"""
import torch
from transformers import AutoModel, AutoProcessor
from typing import List, Optional, Union
from PIL import Image


class CLIPComplaintClassifier:
    """CLIP 기반 민원 분류기"""
    
    def __init__(
        self,
        model_name: str = "Bingsu/clip-vit-base-patch32-ko",
        device: Optional[str] = None,
        temperature: float = 0.07
    ):
        """
        Args:
            model_name: Hugging Face 모델 이름
            device: 사용할 디바이스 (None이면 자동 선택)
            temperature: 온도 파라미터
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.temperature = temperature
        
        print(f"모델 로딩 중: {model_name}")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        print("로딩 완료!")
    
    def classify(
        self,
        labels: List[str],
        image: Optional[Image.Image] = None,
        text: Optional[str] = None
    ) -> torch.Tensor:
        """
        민원 분류
        
        Args:
            labels: 분류할 라벨 리스트
            image: 입력 이미지 (선택)
            text: 입력 텍스트 (선택)
        
        Returns:
            각 라벨에 대한 확률 분포
        """
        if image is None and text is None:
            raise ValueError("이미지와 텍스트 중 적어도 하나는 입력해야 합니다.")
        
        with torch.no_grad():
            if image is not None and text is not None:
                all_texts = [text] + labels
                inputs = self.processor(
                    text=all_texts,
                    images=image,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                outputs = self.model(**inputs)
                
                if hasattr(outputs, 'image_embeds') and hasattr(outputs, 'text_embeds'):
                    image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                    text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                    input_text_embed = text_embeds[0:1]
                    label_embeds = text_embeds[1:]
                    combined_embed = (image_embeds + input_text_embed) / 2
                    combined_embed = combined_embed / combined_embed.norm(dim=-1, keepdim=True)
                    logits = combined_embed @ label_embeds.T
                elif hasattr(outputs, 'logits_per_image'):
                    logits = outputs.logits_per_image
                else:
                    raise ValueError(f"모델 출력 구조를 확인하세요. 출력 타입: {type(outputs)}")
                
            elif image is not None:
                inputs = self.processor(
                    text=labels,
                    images=image,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                outputs = self.model(**inputs)
                
                if hasattr(outputs, 'image_embeds') and hasattr(outputs, 'text_embeds'):
                    image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                    text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                    logits = image_embeds @ text_embeds.T
                elif hasattr(outputs, 'logits_per_image'):
                    logits = outputs.logits_per_image
                else:
                    raise ValueError(f"모델 출력 구조를 확인하세요. 출력 타입: {type(outputs)}")
                
            else:  # text only
                all_texts = [text] + labels
                inputs = self.processor(
                    text=all_texts,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                outputs = self.model(**inputs)
                
                if hasattr(outputs, 'text_embeds'):
                    text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                    input_text_embed = text_embeds[0:1]
                    label_embeds = text_embeds[1:]
                    logits = input_text_embed @ label_embeds.T
                elif hasattr(outputs, 'logits_per_text'):
                    logits = outputs.logits_per_text
                else:
                    raise ValueError(f"모델 출력 구조를 확인하세요. 출력 타입: {type(outputs)}")
            
            probs = (logits / self.temperature).softmax(dim=-1)
            return probs
    
    def predict_top_k(
        self,
        labels: List[str],
        image: Optional[Image.Image] = None,
        text: Optional[str] = None,
        k: int = 5
    ) -> List[tuple]:
        """
        상위 k개 예측 반환
        
        Args:
            labels: 분류할 라벨 리스트
            image: 입력 이미지 (선택)
            text: 입력 텍스트 (선택)
            k: 반환할 상위 개수
        
        Returns:
            [(라벨, 확률), ...] 리스트
        """
        probs = self.classify(labels, image, text)
        top_probs, top_indices = torch.topk(probs[0], k)
        
        results = [
            (labels[idx.item()], prob.item())
            for prob, idx in zip(top_probs, top_indices)
        ]
        return results

