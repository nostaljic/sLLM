import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.multiprocessing as mp
import argparse

def setup(rank, world_size):
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=world_size, rank=rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, ngpus_per_node, args):
    setup(rank, args.world_size)

    # 모델과 토크나이저 로드
    repo = "MarkrAI/RAG-KO-Mixtral-7Bx2-v2.1"
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, return_dict=True, torch_dtype=torch.float16)
    model.cuda(rank)
    model = DDP(model, device_ids=[rank])

    # 입력 준비
    prompt = '''주어진 정보를 기반으로 질문에 답하세요. 답을 모른다면 답을 지어내지 말고 그냥 모른다고 말하세요. 정보: 질문: 호모 에렉투스가 처음으로 언어를 사용하기 시작한 정확한 시점은 언제인가요? 답변: '''
    inputs = tokenizer(prompt, return_tensors="pt").to(rank)

    # 모델 실행
    with torch.no_grad():
        generated_ids = model.module.generate(
            **inputs,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=400,
            do_sample=False,
            num_beams=1,
        )
    
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    if rank == 0:  # 메인 프로세스에서만 결과 출력
        print(outputs[0])

    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(train, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

if __name__ == "__main__":
    main()
