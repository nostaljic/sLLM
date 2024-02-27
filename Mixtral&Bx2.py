def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    # 모델과 토크나이저를 로드합니다.
    repo = "MarkrAI/RAG-KO-Mixtral-7Bx2-v2.1"
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, return_dict=True, torch_dtype=torch.float16)
    model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    # 입력을 준비합니다.
    prompt = '''주어진 정보를 기반으로 질문에 답하세요. 답을 모른다면 답을 지어내지 말고 그냥 모른다고 말하세요. 정보: 질문: 호모 에렉투스가 처음으로 언어를 사용하기 시작한 정확한 시점은 언제인가요? 답변: '''
    inputs = tokenizer(prompt, return_tensors="pt").to(rank)
    
    # 모델을 실행합니다.
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
    print(outputs[0])
    
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # 사용 가능한 GPU 수
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
