import numpy as np
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from sentence_transformers import SentenceTransformer


class Scraper:
    DRIVER_PATH = "./geckodriver"

    def __init__(self, timeout=30, headless=False):
        self.timeout = timeout
        self.headless = headless
        self.driver = self._setup_driver()

    def _setup_driver(self):
        options = FirefoxOptions()
        if self.headless:
            options.add_argument("--headless")

        driver = webdriver.Firefox(options=options, service=Service(self.DRIVER_PATH))
        driver.set_page_load_timeout(self.timeout)
        driver.set_script_timeout(self.timeout)
        return driver

    def fetch_website_source(self, url: str):
        if not self.driver:
            self.driver = self._setup_driver()
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, self.timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            page_source = self.driver.page_source
            return page_source
        except Exception as e:
            print(f"Error loading {url}: {e}")
            raise Exception(
                f"Failed to fetch the website source for {url}. Error: {e}"
            ) from e

    def parse_content(self, page_source):
        try:
            soup = BeautifulSoup(page_source, "html.parser")

            # Find all <p> tags and then all <a> tags within them
            nav_links = {
                a.get_text(): a.get("href")
                for p in soup.find_all("p")
                for a in p.find_all("a")
                if a.get("href") and "cite" not in a.get("href")
            }

            return nav_links

        except Exception as e:
            print(f"Error parsing content: {str(e)}")
            return {}

    def get_current_url(self):
        return self.driver.current_url


BASE_URL = "https://en.wikipedia.org"
MODEL_PATH = "nomic-ai/nomic-embed-text-v1"

import time


class WikiGame:
    def __init__(self, start_url, end_url, batch_size=10):
        self.start_url = start_url
        self.end_url = end_url
        self.scraper = Scraper()
        self.visited = []
        self.batch_size = batch_size

        # Initialize the model for the main thread
        self.model = SentenceTransformer(MODEL_PATH, trust_remote_code=True)
        self.target_embedding = self._compute_embedding(end_url.split(".org")[-1])

    def start_game(self):
        self.visited.append(self.start_url)
        page_source = self.scraper.fetch_website_source(self.start_url)
        nav_links = self.scraper.parse_content(page_source)
        start = time.time()
        self.recurse(nav_links)
        end = time.time()
        print(f"Time: {end-start}")

    def recurse(self, nav_links):
        if self.end_url.replace(BASE_URL, "") in nav_links.values():
            print(f"Fetching: {self.end_url}")
            self.scraper.fetch_website_source(self.end_url)
            print("Reached the end URL!")
            return True

        if self.scraper.get_current_url() == self.end_url:
            print("Reached the target URL!")
            return True

        similarity, link = self.most_similar(nav_links)
        print(f"Fetching: {link}  Similarity: {similarity}")
        full_link = BASE_URL + link

        if link not in self.visited:
            self.visited.append(link)
            page_source = self.scraper.fetch_website_source(full_link)
            fetched_nav_links = self.scraper.parse_content(page_source)
            if self.recurse(fetched_nav_links):
                return True
        return False

    def _compute_embedding(self, text_content: str):
        return self.model.encode(f"classification: {text_content}")

    def compute_embeddings_for_nav_links(self, nav_links):
        links = list(nav_links.values())
        embeddings = self.model.encode([f"classification: {link}" for link in links])
        embeddings_dict = {
            link: embedding for link, embedding in zip(links, embeddings)
        }
        return embeddings_dict

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            print(f"Shape mismatch: {a.shape} vs {b.shape}")
            return 0.0
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def remove_visited_nav_links(self, nav_links):
        d = {k: v for k, v in nav_links.items() if v not in self.visited}
        return d

    def most_similar(self, nav_links):
        nav_links = self.remove_visited_nav_links(nav_links)
        nav_embeddings = self.compute_embeddings_for_nav_links(nav_links)
        most = (0, "")
        for link, embedding in nav_embeddings.items():
            similarity = self.cosine_similarity(self.target_embedding, embedding)
            if similarity >= most[0]:
                most = (similarity, link)
        return most


if __name__ == "__main__":
    import sys

    start_url = sys.argv[1]
    end_url = sys.argv[2]
    game = WikiGame(start_url, end_url, 10)
    game.start_game()
