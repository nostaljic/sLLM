from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import torch


# 모델과 토크나이저 로드
model_name = "hfl/chinese-llama-2-1.3b"  # 예시로 사용할 모델명을 입력하세요
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 데이터셋 로드
dataset_name ="royboy0416/ko-alpaca"
dataset = load_dataset(dataset_name)

# 데이터셋 전처리
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 다중 GPU 사용 설정
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./chinese-llama-2-1.3b",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

import os
os.environ["WANDB_MODE"] = "dryrun"

# 트레이너 생성 및 학습 시작
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
)

trainer.train()

# 모델 저장
trainer.save_model("./chinese-llama-2-1.3b/model")