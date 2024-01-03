import speech_recognition as sr
import webbrowser
import openai
import win32com.client as wn
speaker= wn.Dispatch("SAPI.SpVoice")
def say(text):
    speaker.Speak(text)

def takecommand():
    r=sr.Recognizer()
    with sr.Microphone() as source:
        r.pause_threshold=0.6
        audio=r.listen(source)
        try:
            query=r.recognize_google(audio,language="en-in")
            print(f"user said :{query}")
            return query
        except Exception as e:
            return "some error occured, sorry from Jarvis, try again please"


say("hello user . I am JARVIS AI ")
while True:
    print("listening.....")
    text=takecommand()
    say(text)
    sites=[["Youtube","https://www.youtube.com"],["Wikipedia","https://www.wikipedia.com"],["Google","https://www.google.com"],["Linkedin","https://www.linkedin.com"]]
    for site in sites:
        if f" Open {site[0]}".lower() in text.lower():
            print("ok")
            say(f"opening {site[0]} sir...")
            webbrowser.open(site[1])
        if "open resume" in text:
            path=r"E:\MY IMP DOCS\Resume_tanishqa_allLinks"
            os.system(f"open {path}")

