from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path

model_path = Path("C:/Users/pro/PycharmProjects/bigman/tune/bart_cause_effect_model")

tokenizer = AutoTokenizer.from_pretrained(str(model_path))
model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))

input_text = "US District Judge James Boasberg ruled Wednesday that “probable cause exists” to hold Trump administration officials in criminal contempt for violating his orders in mid-March halting the use of the Alien Enemies Act to deport alleged Venezuelan gang members.The judge is still deciding punishment and the next steps he may take, and is giving the Justice Department an opportunity to respond.The situation has been a major political and legal flashpoint for the Trump White House in its efforts to carry out a historic deportation campaign, especially in mid-March when it sent three planes of migrants to a prison in El Salvador. | cause_time: 2020-11-05 | effect_time: 2021-01-06"
inputs = tokenizer(input_text, return_tensors="pt")

output = model.generate(**inputs, max_new_tokens=100)
print("🧠 예측 결과:")
print(tokenizer.decode(output[0], skip_special_tokens=True))
