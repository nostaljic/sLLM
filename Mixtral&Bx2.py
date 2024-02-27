from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# CUDA 사용 가능한 GPU가 있는지 확인하고, 사용 가능한 GPU 개수를 확인합니다.
if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()
    print(f"Number of GPUs available: {n_gpu}")
    device = torch.device("cuda")
else:
    raise SystemError("CUDA is not available. This script requires a GPU environment.")

repo = "MarkrAI/RAG-KO-Mixtral-7Bx2-v2.1"
markrAI_RAG_tokenizer = AutoTokenizer.from_pretrained(repo)

# 모델을 불러오고 DataParallel을 사용해 모델을 여러 GPU에 분배합니다.
markrAI_RAG = AutoModelForCausalLM.from_pretrained(
    repo,
    return_dict=True,
    torch_dtype=torch.float16
)
markrAI_RAG = torch.nn.DataParallel(markrAI_RAG).to(device)

prompt = '''
주어진 정보를 기반으로 질문에 답하세요. 답을 모른다면 답을 지어내지 말고 그냥 모른다고 말하세요.
정보:

질문:

답변:
'''

# GPU로 데이터를 보냅니다.
inputs = markrAI_RAG_tokenizer(prompt, return_tensors="pt").to(device)
generated_ids = markrAI_RAG.module.generate(
    **inputs,
    num_return_sequences=1,
    eos_token_id=markrAI_RAG_tokenizer.eos_token_id,
    pad_token_id=markrAI_RAG_tokenizer.pad_token_id,
    max_new_tokens=400,
    do_sample=False,
    num_beams=1,
)

# 결과를 CPU로 옮기고 디코딩합니다.
outputs = markrAI_RAG_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(outputs[0])
