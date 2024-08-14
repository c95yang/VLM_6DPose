import torch
from tqdm import tqdm
from torchvision.transforms import ToPILImage
from transformers import BitsAndBytesConfig, pipeline
import json
from misc import parse_output

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from datasets import Remote60_seq
from PIL import Image 
from matplotlib import pyplot as plt

# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForPreTraining

# def generate_descriptions(image_paths, prompt, train_dataset, val_dataset, topil):
#     processor = AutoProcessor.from_pretrained("llava-hf/bakLlava-v1-hf")
#     model = AutoModelForPreTraining.from_pretrained("llava-hf/bakLlava-v1-hf")

#     for image_path in tqdm(image_paths, desc="Generating train descriptions"):
#         print(image_path)
#         image = train_dataset.load_image(image_path)  
#         image = topil(image)
#         inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
#         output = model.generate(**inputs, max_new_tokens=100)
#         print(processor.decode(output[0], skip_special_tokens=True))

#         #description = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 77})
#         description_text = parse_output(description[0]["generated_text"]) 
#         train_descriptions[image_path] = description_text

#     with open("train_descriptions.json", "w") as f:
#         json.dump(train_descriptions, f)

#     val_descriptions = {}
#     image_paths = val_dataset.get_all_image_paths()

#     for image_path in tqdm(image_paths, desc="Generating val descriptions"):
#         image = train_dataset.load_image(image_path)  
#         image = topil(image)
#         description = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 77})
#         description_text = parse_output(description[0]["generated_text"])  
#         val_descriptions[image_path] = description_text

#     with open("val_descriptions.json", "w") as f:
#         json.dump(val_descriptions, f)

def generate_descriptions_pipe(prompt, dataset, topil, path):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    pipe = pipeline("image-to-text", model=path, model_kwargs={"quantization_config": quantization_config}) 

    # # Test the pipeline
    # with torch.no_grad(): 
    #     img = Image.open("data/remote60/train/remote-comfee/render_position((-0.397396333583049, -0.12500000000000003, 0.2764980181750858))_rotation(<Euler (x=0.9848, y=0.0000, z=-1.2660), order='XYZ'>).png")
    #     description = pipe(img, prompt=prompt, generate_kwargs={"max_new_tokens": 77})
    #     # print(pipe.model)
    #     description = parse_output(description[0]["generated_text"]) 
    #     print(description)
    #     return

    descriptions = {}
    image_paths = dataset.get_all_image_paths()
    
    for image_path in tqdm(image_paths, desc="Generating train descriptions"):
        image = dataset.load_image(image_path)  
        image = topil(image)
        # plt.imshow(image)
        # plt.show()
        description = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 77})
        description_text = parse_output(description[0]["generated_text"]) 
        descriptions[image_path] = description_text
        # print(description_text)

    with open("descriptions/descriptions.json", "w") as f:
        json.dump(descriptions, f)


if __name__ == '__main__':

    question = "Describe the romote in the image, not the background and hand. For example, you can describe the orientation."

    prompt = "USER: <image>\n" + question + "\nASSISTANT:"
    topil = ToPILImage()

    image_dir = 'data/remote60'
    # train_dataset = Remote60_seq(root_dir=image_dir, is_train=True)
    # val_dataset = Remote60_seq(root_dir=image_dir, is_val=True)
    test_dataset = Remote60_seq(root_dir=image_dir, is_test=True)

    generate_descriptions_pipe(prompt, test_dataset, topil, 
                             "llava-hf/llava-1.5-7b-hf") #"llava-hf/bakLlava-v1-hf","llava-hf/llava-1.5-7b-hf"