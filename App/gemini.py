import google.generativeai as genai

genai.configure(api_key="AIzaSyB82tyQUt9TaB1p15GqtI-_8t9CIBUn-Bk")

model = genai.GenerativeModel("models/gemini-1.5-flash")
response = model.generate_content("Hello! What can you do?")
print(response.text)