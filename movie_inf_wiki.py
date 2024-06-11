import requests
from bs4 import BeautifulSoup
import re

# URL of the page to scrape
url = 'https://en.wikipedia.org/wiki/Academy_Award_for_Best_Picture'  # 请替换为实际的URL

film_url={}
film_length={}

response = requests.get(url)
response.encoding = 'utf-8'  # 确保正确的编码

soup = BeautifulSoup(response.text, 'html.parser')


section_1950s = soup.find('span', id='1950s').parent

table = section_1950s.find_next('table')

rows=table.find_all('tr')
for row in rows:
    cells=row.find_all("td")
    for cell in cells:
        lines=cell.find_all('i')
        for line in lines:
            a_tag=line.find('a')
            if a_tag:
                href = a_tag.get('href')
                title = a_tag.get('title')
                film_url[title]='https://en.wikipedia.org/'+str(href)
               
           
for item in film_url:
    response=requests.get(film_url[item])
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, 'html.parser')
    info_box = soup.find('table', {'class': 'infobox vevent'})
    runtime_row = info_box.find('th', string='Running time').find_next_sibling('td')
    length=runtime_row.get_text(strip=True)

    pattern = re.compile(r'\[.*?\]')
    if '[' in length:
        length = pattern.sub('', length)
    film_length[item]=length
print(film_length)




# response=requests.get(film_url['All About Eve'])
# response.encoding = 'utf-8'
# soup = BeautifulSoup(response.text, 'html.parser')
# info_box = soup.find('table', {'class': 'infobox vevent'})
# runtime_row = info_box.find('th', string='Running time').find_next_sibling('td')
# length=runtime_row.get_text(strip=True)

# pattern = re.compile(r'\[.*?\]')
# if '[' in length:
#     length = pattern.sub('', length)
# film_length['x']=length
# print(film_length)
