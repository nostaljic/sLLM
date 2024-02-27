import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer

def main(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    # 모델과 토크나이저 로드
    repo = "MarkrAI/RAG-KO-Mixtral-7Bx2-v2.1"
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, return_dict=True, torch_dtype=torch.float16)
    
    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # 여기에 모델을 사용하는 코드 추가
    # 예: 모델로부터 출력을 생성하고 출력을 인쇄
    prompt = "Your prompt here"
    inputs = tokenizer(prompt, return_tensors="pt").to(rank)
    with torch.no_grad():
        generated_ids = ddp_model.module.generate(
            **inputs,
            max_length=50,  # 출력의 최대 길이 설정
            num_beams=5,    # 빔 서치의 빔 수 설정
            temperature=1.0 # 샘플링 온도 설정
        )
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(generated_texts)

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    world_size = 2  # 사용할 GPU 수
    torch.multiprocessing.spawn(main,
                                args=(world_size,),
                                nprocs=world_size,
                                join=True)
