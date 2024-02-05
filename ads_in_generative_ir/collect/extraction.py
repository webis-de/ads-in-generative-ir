import argparse
import csv
import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from random import shuffle, uniform
import time
from tqdm import tqdm
from typing import Sequence, Tuple

from ads_in_generative_ir import RESOURCE_PATH

class Extractor:
    def __init__(self, out_file: str, queries_file: str):
        self.out_file = RESOURCE_PATH / f'generated_responses/{out_file}'
        self.driver = webdriver.Firefox()

        # Create headers if the out_file does not exist
        if not self.out_file.exists():
            with open(self.out_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(['service', 'query', 'response', 'sources'])

        # Load the queries from a txt file (expecting one query per line)
        with open(RESOURCE_PATH / f'queries/{queries_file}', 'r') as file:
            self.queries = [l.replace('\n', '') for l in file.readlines()]
        shuffle(self.queries)

    def collect_responses(self, service: str = "youchat"):
        """
        Iterate over all strings in self.queries and collect the responses for the provided chat service.
        Results are written to self.out_file.
        :param service:     Name of the service to collect responses for
        """
        match service:
            case "bing":
                response_func = self.get_bing_response
            case "youchat":
                response_func = self.get_youchat_response
            case _:
                print("Please provide a valid search service")
                return

        with open(self.out_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')

            for q in tqdm(self.queries, desc=f'Collecting responses from {service}'):
                try:
                    response, sources = response_func(query=q)
                except:
                    print(f"Failed to get results for '{q}' and service '{service}'.")
                    continue

                writer.writerow([service, q, response, sources])
                time.sleep(uniform(1.5, 3))

        self.driver.close()

    def get_youchat_response(self, query: str) -> Tuple[str, Sequence[Tuple]]:
        """
        Get the response and sources used in the response for you.com
        :param query:   Query for which to collect a response
        :return:        response    - String containing the response made by the chatbot
                        sources     - List of source tuples used in the response ("name", "number", "link")
        """
        # (Re-)load the page and wait until the input form is visible
        self.driver.get('https://you.com/')
        input_wait = WebDriverWait(self.driver, timeout=20)
        input_form = input_wait.until(EC.presence_of_element_located((By.ID, "search-input-textarea")))

        # Input the query with random pauses between words
        self._delayed_typing(element=input_form, text=query)
        input_form.send_keys(Keys.ENTER)

        # Detect if the response is complete by looking for the divs with data-testid="suggest-chat-chip"
        response_wait = WebDriverWait(self.driver, timeout=60)
        suggest_div = response_wait.until(
            EC.presence_of_element_located((By.XPATH, "//div[@data-testid='suggested-chat-chip']")))

        # Identify the response div by looking for paragraphs with "data-testid='youchat-text'" in the chatHistory Div
        # The parent of these paragraphs is the div, we're looking for
        chat_history = self.driver.find_element(By.ID, "chatHistory")
        response_div = chat_history.find_element(By.XPATH, ".//p[@data-testid='youchat-text']/parent::div")

        # A div at the end of the responseDiv might contain source links. These should be returned separately
        sources = []
        child_nodes = response_div.find_elements(By.XPATH, "./*")
        if child_nodes[-1].tag_name == 'div':
            source_div = child_nodes[-1]
            child_nodes = child_nodes[:-1]

            # Create a list of sources; Tuples with ("name", "number", "link")
            try:
                source_wait = WebDriverWait(self.driver, timeout=5)
                tmp = source_wait.until(
                    EC.presence_of_element_located((By.XPATH, "//a[@data-eventappname='click_on_citation']")))
                source_links = source_div.find_elements(By.XPATH, "./a[@data-eventappname='click_on_citation']")
                sources = [tuple([*s.text.split("\n"), s.get_attribute("href")]) for s in source_links]
            except:
                pass

        response = "\n".join([child.text for child in child_nodes])
        return response, sources

    def get_bing_response(self, query: str) -> Tuple[str, Sequence[Tuple]]:
        """
        Get the response and sources used in the response for bing.com's chatbot
        :param query:   Query for which to collect a response
        :return:        response    - String containing the response made by the chatbot
                        sources     - List of source tuples used in the response ("name", "number", "link")
        """
        # (Re-)load the page and click the reject cookies button (if it is there)
        self.driver.get('https://www.bing.com/search?q=Bing+AI&showconv=1')
        self.driver.implicitly_wait(uniform(1.5, 3))

        try:
            btn = self.driver.find_element(By.ID, "bnp_btn_reject")
            btn.click()
        except:
            pass

        # Wait until the top most shadow root is visible
        input_wait = WebDriverWait(self.driver, timeout=20)
        main_elem = input_wait.until(EC.presence_of_element_located((By.CLASS_NAME, "cib-serp-main")))
        sr_1 = main_elem.shadow_root
        input_form = (sr_1.find_element(By.ID, 'cib-action-bar-main').shadow_root
                      .find_element(By.CLASS_NAME, "main-container")
                      .find_element(By.CLASS_NAME, "input-row")
                      .find_element(By.TAG_NAME, "cib-text-input").shadow_root
                      .find_element(By.ID, "searchbox"))

        # Input the query with random pauses between words
        self._delayed_typing(element=input_form, text=query)
        input_form.send_keys(Keys.ENTER)

        # Detect if the response is complete by looking for the suggestion bar
        sr_2 = sr_1.find_element(By.ID, "cib-conversation-main").shadow_root
        suggestion_bar = sr_2.find_element(By.CSS_SELECTOR, "cib-suggestion-bar").shadow_root
        i = 0
        while True:
            self.driver.implicitly_wait(1)
            try:
                items = suggestion_bar.find_elements(By.CSS_SELECTOR, "cib-suggestion-item")
                if len(items) > 0:
                    break
            except Exception as e:
                pass

            if i > 59:
                raise RuntimeError('Suggestion Bar for Bing was not found in time')
            i += 1

        # Get the response and sources
        sr_3 = (sr_2.find_element(By.ID, "cib-chat-main")
                .find_element(By.TAG_NAME, "cib-chat-turn").shadow_root
                .find_element(By.CLASS_NAME, "response-message-group").shadow_root
                .find_element(By.CSS_SELECTOR, "cib-message[type='text']").shadow_root)

        response = sr_3.find_element(By.CLASS_NAME, "content").text

        source_links = (sr_3.find_element(By.CSS_SELECTOR, "cib-message-attributions").shadow_root
                        .find_elements(By.CLASS_NAME, "attribution-item"))

        # Create a list of sources; Tuples with ("name", "number", "link")
        sources = [tuple([*[x for x in reversed(s.text.split(". "))], s.get_attribute("href")]) for s in source_links]

        return response, sources

    @staticmethod
    def _delayed_typing(element: WebElement, text: str, min_delay: float = 0.01, max_delay: float = 0.15):
        for c in text:
            element.send_keys(c)
            time.sleep(uniform(min_delay, max_delay))



class ExtractionRunner:
    def __init__(self, service: str, meta_topic: str):
        self.service = service
        self.meta_topic = meta_topic
        self.all_queries_file = RESOURCE_PATH / f'queries/{meta_topic}_queries.txt'
        self.new_queries_file = RESOURCE_PATH / f'queries/{meta_topic}_queries_{service}.txt'

        self._create_dir()

    def _create_dir(self):
        out_dir = RESOURCE_PATH / "generated_responses"
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    def update_new_queries(self, first_run=True, min_counts=2):
        # Get the list of all queries
        with open(self.all_queries_file, 'r') as file:
            queries = [l.replace('\n', '') for l in file.readlines()]

        try:
            df = pd.read_csv(RESOURCE_PATH / f'generated_responses/{self.meta_topic}_responses.csv')
            df = df.loc[df['service'] == self.service]

            if first_run:
                processed_queries = df['query'].unique().tolist()
                new_queries = [q for q in queries if q not in processed_queries]

            else:
                counts = df.groupby('query')['service'].count()
                new_queries = counts.loc[counts < min_counts].index.tolist()

        except:
            new_queries = queries

        print(f'Queries left: {len(new_queries)}')

        with open(self.new_queries_file, 'w') as file:
            for q in new_queries:
                file.write(f'{q}\n')

        return new_queries

    def run(self, num_queries=30, first_run=True, min_counts=2, max_retries=3):
        # Update the list of new queries
        new_queries = self.update_new_queries(first_run, min_counts)
        prev_len = len(new_queries)
        retries = 0

        # Loop over the new queries
        finished = False
        while not finished:
            # Collect 100 queries at a time
            e = Extractor(out_file=f'{self.meta_topic}_responses.csv',
                          queries_file=f'{self.meta_topic}_queries_{self.service}.txt')
            e.queries = e.queries[:num_queries]
            e.collect_responses(service=self.service)

            # Update new queries depening on the run and min_counts
            new_queries = self.update_new_queries(first_run, min_counts)

            print(f'Queries left: {len(new_queries)}')
            if len(new_queries) == 0 or retries >= max_retries:
                finished = True
                
            else:
                if len(new_queries) == prev_len:
                    retries += 1
                prev_len = len(new_queries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Response collection',
        description='Collect responses for a meta topic. '
                    'Queries are taken from the corresponding file in the resources folder')

    parser.add_argument("meta_topic", metavar="M", type=str,
                        help='Name of the meta topic to collect responses for.')
    parser.add_argument('-s', '--service', type=str, default='bing',
                        help='Name of the conversational search engine from which to collect responses.')
    parser.add_argument('--additional_run', action=argparse.BooleanOptionalAction,
                        help='To obtain more than one response per query, set this flag.')
    parser.add_argument('-c', '--counts', type=int, default=2,
                        help='Number of responses to collect per query. Only relevant if not the first run.')
    args = parser.parse_args()

    topic = args.meta_topic
    service = args.service
    counts = args.counts
    first_run = False if args.additional_run else True

    num_queries = 30 if service == 'bing' else 100

    er = ExtractionRunner(service=service, meta_topic=topic)
    er.run(num_queries=30, first_run=first_run, min_counts=counts)
