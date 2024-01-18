import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

st.title('Chatbot')

# GPT2모델 로드
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

input_sentence = st.text_input("You: ", "")

if st.button("Reply"):
    # 사용자의 입력을 모델에 제공
    input_ids = tokenizer.encode(input_sentence, return_tensors='pt')
    generated_output = model.generate(input_ids, max_length=100, temperature=0.7, num_return_sequences=1)

    # 생성된 답변 출력
    generated_sentence = tokenizer.decode(generated_output[0], skip_special_tokens=True)
    st.text("Bot: " + generated_sentence)
