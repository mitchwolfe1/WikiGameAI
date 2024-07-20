import time

from src.embeddings import Embeddings
from src.scraper import Scraper
from src.util import remove_visited_nav_links

BASE_URL = "https://en.wikipedia.org"


class WikiGame:
    def __init__(self, start_url, end_url, batch_size=10):
        self.start_url = start_url
        self.end_url = end_url
        self.scraper = Scraper()
        self.visited = []
        self.batch_size = batch_size
        self.embeddings = Embeddings()
        self.target_embedding = self.embeddings.compute_embedding(
            end_url.split(".org")[-1]
        )
        self.num_hops = 0

    def start_game(self):
        nav_links = self.visit_page(self.start_url)

        start = time.time()
        self.recurse(nav_links)
        end = time.time()
        print(f"Num hops: {self.num_hops}")
        print(f"Time: {end-start}")

    def recurse(self, nav_links):
        if (
            self.scraper.get_current_url() == self.end_url
            or self.end_url.replace(BASE_URL, "") in nav_links.values()
        ):
            print(f"Fetching: {self.end_url}")
            self.scraper.fetch_website_source(self.end_url)
            print("Reached the end URL!")
            return True

        similarity, link = self.most_similar(nav_links)
        print(f"Fetching: {link}  Similarity: {similarity}")
        full_link = BASE_URL + link

        if link not in self.visited:
            fetched_nav_links = self.visit_page(full_link)
            self.num_hops += 1
            if self.recurse(fetched_nav_links):
                return True
        return False

    def visit_page(self, link):
        self.visited.append(link.split(".org")[-1])
        page_source = self.scraper.fetch_website_source(link)
        nav_links = self.scraper.parse_content(page_source)
        if len(nav_links) == 0:
            print("Reached dead end... going back")
            return self.visit_page(
                self.visited[-2]
            )  # If we reach a dead end, go back and try again
        else:
            nav_links = {
                key: value
                for key, value in nav_links.items()
                if value.startswith("/wiki/")
            }
            return nav_links

    def most_similar(self, nav_links):
        nav_links = remove_visited_nav_links(nav_links, self.visited)
        nav_embeddings = self.embeddings.compute_embeddings_for_nav_links(nav_links)
        most = (0, "")
        for link, embedding in nav_embeddings.items():
            similarity = self.embeddings.cosine_similarity(
                self.target_embedding, embedding
            )
            if similarity >= most[0]:
                most = (similarity, link)
        return most
