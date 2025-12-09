import abc
import time
import random
import logging
import requests
from typing import Any, Dict, List, Optional, Union
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, urlparse
import json


class Url_CrawlBase(abc.ABC):
    """
    Base crawler class
    Provides a generic crawling framework that supports various data sources such as web pages, APIs, and PDFs
    """

    def __init__(
        self,
        base_url: str = "",
        headers: Optional[Dict[str, str]] = None,
        delay: float = 1.0,
        max_retries: int = 3,
        timeout: int = 10,
        respect_robots: bool = True,
        proxy: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize crawler configuration

        Parameters:
            base_url: Base URL (for resolving relative paths)
            headers: Request headers dictionary
            delay: Request interval (seconds)
            max_retries: Maximum retry attempts
            timeout: Timeout period (seconds)
            respect_robots: Whether to respect robots.txt
            proxy: Proxy address (e.g.: http://127.0.0.1:8080)
            logger: Logger instance (optional)
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(headers or self._default_headers())
        self.timeout = timeout
        if proxy:
            self.session.proxies.update({"http": proxy, "https": proxy})

        self.delay = delay
        self.max_retries = max_retries
        self.respect_robots = respect_robots
        self.logger = logger or self._setup_logger()

        # Configure robots.txt parser
        self.robot_parser = RobotFileParser()
        if base_url and respect_robots:
            self._parse_robots()

    def _default_headers(self) -> Dict[str, str]:
        """Default request headers"""
        return {
            "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Mobile Safari/537.36"
        }

    def _setup_logger(self) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _parse_robots(self):
        """Parse robots.txt"""
        robots_url = urljoin(self.base_url, "/robots.txt")
        try:
            self.robot_parser.set_url(robots_url)
            self.robot_parser.read()
            self.logger.info(f"✅ Successfully parsed robots.txt: {robots_url}")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to parse robots.txt: {e}")

    def _can_fetch(self, url: str) -> bool:
        """Check if robots.txt allows crawling"""
        if not self.respect_robots:
            return True
        try:
            return self.robot_parser.can_fetch("*", url)
        except:
            return True

    def _rate_limit(self):
        """Rate limiting"""
        self.logger.debug(f"Request delay: {self.delay}s")
        time.sleep(self.delay)

    @abc.abstractmethod
    def fetch(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Abstract method: Fetch raw data
        Subclasses must implement this method
        """
        pass

    @abc.abstractmethod
    def parse(self, raw_data: Any) -> List[Dict[str, Any]]:
        """
        Abstract method: Parse raw data
        Returns a list of structured data
        """
        pass

    def store(
        self,
        parsed_data: List[Dict[str, Any]],
        vector_store: Optional[Any] = None,
        cleaner: Optional[Any] = None,
    ) -> List[str]:
        """
        Store data (optional)
        :param parsed_data: Parsed data
        :param vector_store: Vector store instance
        :param cleaner: Data cleaner instance
        :return: List of stored IDs
        """
        if not parsed_data:
            self.logger.warning("No data to store")
            return []

        ids = []
        for item in parsed_data:
            # If a cleaner is provided, clean first
            if cleaner:
                try:
                    item = cleaner.clean(item)
                except Exception as e:
                    self.logger.error(f"Cleaning failed: {e}")
                    continue

            # If a vector store is provided, store the data
            if vector_store:
                try:
                    doc_id = vector_store.add_document(
                        text=item.get("content", ""), metadata=item.get("metadata", {})
                    )
                    ids.append(doc_id)
                except Exception as e:
                    self.logger.error(f"Storage failed: {e}")

        self.logger.info(f"Storage completed: {len(ids)} records")
        return ids

    def run(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        vector_store: Optional[Any] = None,
        cleaner: Optional[Any] = None,
    ) -> List[str]:
        """
        Execute the complete crawling workflow: fetch -> parse -> store

        :return: List of stored IDs
        """
        self.logger.info(f"Starting crawl: {url}")

        # 1. Check robots.txt
        if not self._can_fetch(url):
            self.logger.error(f"robots.txt prohibits crawling: {url}")
            return []

        # 2. Rate limiting
        self._rate_limit()

        # 3. Fetch data
        raw_data = self.fetch(url, params)

        # 4. Parse data
        parsed_data = self.parse(raw_data)
        self.logger.info(f"Parsing completed: {len(parsed_data)} records")

        # 5. Store data
        ids = self.store(parsed_data, vector_store, cleaner)

        return ids

    def _request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        Request with retries
        """
        for attempt in range(self.max_retries):
            try:
                response = getattr(self.session, method)(url, **kwargs)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt+1}): {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2**attempt)  # Exponential backoff
