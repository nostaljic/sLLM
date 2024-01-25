from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# 모델과 토크나이저 로드
model_name = "alpaca"  # 예시로 사용할 모델명을 입력하세요
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 데이터셋 로드
dataset_name = "junelee/sharegpt_deepl_ko"
dataset = load_dataset(dataset_name)

# 데이터셋 전처리
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./alpaca_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# 트레이너 생성 및 학습 시작
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
)

trainer.train()

# 모델 저장
trainer.save_model("./alpaca_finetuned")
