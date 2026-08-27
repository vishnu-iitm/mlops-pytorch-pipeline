import torch
from src.model import get_model

def test_model_output_shape():
    # check that it outputs 10 classes as expected
    model = get_model('resnet18', 10)
    model.eval()
    
    dummy_x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = model(dummy_x)
        
    assert out.shape == (2, 10), f"expected shape (2, 10) but got {out.shape}"
