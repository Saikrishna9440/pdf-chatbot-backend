import os
import requests
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"

def fetch_GovtScheme_news(query: str = "Indian Government schemes", page_size: int = 5):
    try:
        if not NEWS_API_KEY:
            return {"error": "⚠️ NEWS_API_KEY not found in environment. Please set it first."}

        params = {
            "q": f"{query}",
            "pageSize": page_size,
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": NEWS_API_KEY,
            # optional: focus on Indian sources only
            "domains": "pib.gov.in,thehindu.com,indiatimes.com,ndtv.com,indiatoday.in,financialexpress.com"
        }

        response = requests.get(NEWS_API_URL, params=params)

        if response.status_code != 200:
            return {"error": f"API request failed with status {response.status_code}: {response.text}"}

        data = response.json()

        if not data.get("articles"):
            return {"error": "No relevant Indian government articles found."}

        articles = []
        for article in data["articles"]:
            # Simple relevance filter (extra safety)
            title = article.get("title", "").lower()
            desc = article.get("description", "").lower()
            if any(word in title + desc for word in ["india", "indian", "government", "scheme", "ministry", "yojana", "pm"]):
                articles.append({
                    "title": article.get("title"),
                    "description": article.get("description", "No description available"),
                    "url": article.get("url"),
                    "source": article["source"].get("name", "Unknown")
                })

        if not articles:
            return {"error": "No Indian government-specific news found."}

        return {"news": articles}

    except Exception as e:
        return {"error": str(e)}


# Test the function directly
if __name__ == "__main__":
    result = fetch_GovtScheme_news("Ayushman Bharat")

    if "news" in result:
        print("\n📰 Latest Indian Government Scheme News:\n")
        for i, article in enumerate(result["news"], 1):
            print(f"{i}. {article['title']}")
            print(f"   Source: {article['source']}")
            print(f"   URL: {article['url']}")
            print(f"   Description: {article['description']}\n")
    else:
        print("⚠️ Error:", result["error"])
