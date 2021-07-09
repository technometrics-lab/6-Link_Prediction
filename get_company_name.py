from bs4 import BeautifulSoup
from selenium import webdriver
import json
import networkx as nx
from networkx.readwrite import json_graph
import time

result_path = "indeed_graph/graph201803.json"
with open(result_path, 'r', encoding = 'utf-8') as file:
    data = json.load(file)
    g = json_graph.node_link_graph(data)
browser = webdriver.Firefox()
mapping={}

for comp in list(g):
    company = comp
    if not company.startswith("CH"):
        print(company)
        continue
    url="https://www.zefix.ch/fr/search/entity/list?name="+company+"&searchType=exact"
    browser.get(url)
    time.sleep(1)
    html_source = browser.page_source
    soup = BeautifulSoup(html_source, "lxml")
    res= soup.find_all("div", {"class": "company-name ng-scope"})
    if len(res) == 0:
        continue
    mapping[company] = res[0]["title"]

with open("company_matcher2.json", "w") as fp:
    json.dump(mapping, fp, sort_keys = True, indent = 4)
browser.quit()
