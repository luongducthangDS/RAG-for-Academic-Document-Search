from google import genai

client = genai.Client(api_key="AIzaSyBJzhPeUWCmWY1wSdMBJ0yGMyHG5pnjMpQ")
try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Bạn là gì?"
    )
    print(response.text)
except Exception as e:
    print("Lỗi:", e)