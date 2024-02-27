import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 분산 설정을 위한 람다 함수 정의
def run_ddp(rank, world_size):
    # 분산 환경 설정
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='tcp://localhost:23456',
        world_size=world_size,
        rank=rank
    )

    # 모델과 토크나이저 로드
    repo = "MarkrAI/RAG-KO-Mixtral-7Bx2-v2.1"
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        return_dict=True,
        torch_dtype=torch.float16
    ).to(rank)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # 모델 사용 예제
    prompt = "Your prompt here"
    inputs = tokenizer(prompt, return_tensors="pt").to(rank)
    with torch.no_grad():
        generated_ids = model.module.generate(**inputs, max_length=50, num_beams=5, temperature=1.0)
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    if rank == 0:  # 메인 프로세스에서만 출력
        print(generated_texts)

    # 분산 환경 정리
    torch.distributed.destroy_process_group()

# 메인 함수에서 분산 작업 시작
if __name__ == "__main__":
    world_size = 2  # 사용할 GPU 수
    torch.multiprocessing.spawn(
        lambda rank: run_ddp(rank, world_size),
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
