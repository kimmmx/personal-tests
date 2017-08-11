import requests

base_url = 'http://w2.weather.gov/climate/getclimate.php?wfo=lot'
form_data = {
    'product': 'CF6',
    'station': 'ORD',
    'recent': 'yes'
    }

response = requests.post(base_url, data=form_data)
text = response.text
for line in text.split("\n"):
    print line
