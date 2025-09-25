import google.generativeai as genai

genai.configure(api_key="AIzaSyB82tyQUt9TaB1p15GqtI-_8t9CIBUn-Bk")

model = genai.GenerativeModel("gemini-2.5-pro")

# to know the models which work with our api
"""print("Available models:\n")
for model in genai.list_models():
    print(model.name, "-", model.supported_generation_methods)"""
Prompts_text = "App/prompts.txt"
with open(Prompts_text,"r",encoding="utf-8") as f:
    content = f.read()
prompt=[p.strip() for p in content.split("=== Prompt") if p.strip()]

for i, text in enumerate(prompt,1):
    print("Prompt: ",i)
    response = model.generate_content(text)
    print(response.text)


