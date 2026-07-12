import requests
from bs4 import BeautifulSoup
import re
import time

BASE_URL = "https://en.wikipedia.org"
url = BASE_URL + "/wiki/Academy_Award_for_Best_Picture"

## so that the wikipedia page doesnot request our multiple requests
headers = {
    "User-Agent": "Mozilla/5.0"
}

film_url = {}
film_length = {}

response = requests.get(url, headers=headers)
if response.status_code != 200:
    print("Failed to fetch main page")
    exit()

response.encoding = 'utf-8'
soup = BeautifulSoup(response.text, 'html.parser')

section_1950s = None

for header in soup.find_all(['h2', 'h3']):
    if '1950s' in header.get_text():
        section_1950s = header
        break

if not section_1950s:
    print("1950s section not found")
    exit()

table = section_1950s.find_next('table')
if not table:
    print("Table not found")
    exit()

rows = table.find_all('tr')

for row in rows:
    cells = row.find_all("td")

    for cell in cells:
        for line in cell.find_all('i'):
            a_tag = line.find('a')

            if not a_tag:
                continue

            href = a_tag.get('href')
            title = a_tag.get('title') or a_tag.text

            if not href:
                continue

            film_url[title] = BASE_URL + href

for movie, link in film_url.items():
    print(f"Fetching: {movie}")

    response = requests.get(link, headers=headers)
    if response.status_code != 200:
        film_length[movie] = "N/A"
        continue

    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, 'html.parser')

    info_box = soup.find('table', {'class': 'infobox vevent'})
    if not info_box:
        info_box = soup.find('table', {'class': 'infobox'})

    if not info_box:
        film_length[movie] = "N/A"
        continue

    runtime_header = info_box.find('th', string='Running time')
    if not runtime_header:
        runtime_header = info_box.find('th', string=lambda x: x and 'Running time' in x)

    if not runtime_header:
        film_length[movie] = "N/A"
        continue

    runtime_row = runtime_header.find_next_sibling('td')
    if not runtime_row:
        film_length[movie] = "N/A"
        continue

    length = runtime_row.get_text(strip=True)
    length = re.sub(r'\[.*?\]', '', length)

    film_length[movie] = length if length else "N/A"

    time.sleep(0.5)## this prevents you from getting blocked from the webpage

print("\nFinal Results:\n")
for movie, runtime in film_length.items():
    print(f"{movie}: {runtime}")