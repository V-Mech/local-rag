from app.scraper import scrape_website

url = "https://en.wikipedia.org/wiki/Artificial_intelligence"

text = scrape_website(url)

print(text[:2000])
