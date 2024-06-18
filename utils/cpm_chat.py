# Use python 3.10 !

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

def cpm_chat():
    model = AutoModel.from_pretrained('openbmb/MiniCPM-V', trust_remote_code=True, torch_dtype=torch.float16)
    # For Nvidia GPUs support BF16 (like A100, H100, RTX3090)
    #model = model.to(device='cuda', dtype=torch.bfloat16)
    # For Nvidia GPUs do NOT support BF16 (like V100, T4, RTX2080)
    model = model.to(device='cuda', dtype=torch.float16)
    # For Mac with MPS (Apple silicon or AMD GPUs).
    # Run with `PYTORCH_ENABLE_MPS_FALLBACK=1 python test.py`
    #model = model.to(device='mps', dtype=torch.float16)

    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V', trust_remote_code=True)
    model.eval()

    image = Image.open('renders/remote14/remote-black/BottomLeftBack.png').convert('RGB')
    question = "Tell me from which direction is the remote in the image observed, \
    using options such as front, back, left, right, top, bottom, and their combinations. Provide several options."
    msgs = [{'role': 'user', 'content': question}]

    res, context, _ = model.chat(
        image=image,
        msgs=msgs,
        context=None,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.7
    )
    print(context)
    print(res)

if __name__ == '__main__':
    cpm_chat()