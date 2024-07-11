# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

linear_projection = torch.nn.Linear(image_features.size(-1), 4096).to(model_class.device)
projected_image_features = linear_projection(image_features)

image_features_str = [",".join(map(str, feature.tolist())) for feature in projected_image_features]
combined_inputs = [f"{desc} [IMG] {img_feat}" for desc, img_feat in zip(descriptions, image_features_str)]

llm_tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
llm_model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5").to(model_class.device)
llm_model.eval()
tokens = llm_tokenizer(combined_inputs, return_tensors="pt", padding=True).to(model_class.device)
print(tokens.shape)
outputs = llm_model(**tokens)