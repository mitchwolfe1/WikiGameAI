from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


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
