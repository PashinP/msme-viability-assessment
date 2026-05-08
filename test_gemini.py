import google.generativeai as genai
genai.configure(api_key="AIzaSyBRwC765H66n1M04N9ODgqERWeWluZbCjE")
model = genai.GenerativeModel("gemini-flash-latest")
try:
    response = model.generate_content("hello")
    print(response.text)
except Exception as e:
    print("ERROR:", repr(e))
    # Print the class name of the error and full details
    import traceback
    print(traceback.format_exc())
