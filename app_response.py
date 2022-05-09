import requests
import sys
num = sys.argv[1]
# resp = requests.post("http://localhost:5000/", files={'file': open(f'{num}.png', 'rb')})
resp = requests.post("https://mnistprediction-zpiw6hekja-uc.a.run.app", files={'file': open(f'{num}.png', 'rb')})

print(resp.json())