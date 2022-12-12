from logger import Logger
import requests
import typing
from bs4 import BeautifulSoup
import os

logger = Logger()

def scrape_page(url: str) -> typing.List[str]:
    """scrape paragraph data from a given url."""
    logger.log_info(f"trying to scrape url='{url}'")
    
    raw_lines: typing.List[str] = []
    try:
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        
        # TODO(Sean) strip some paragraph tags that don't pertain to the transcript
        p_tags = soup.find_all("p")
        i = 0
        for paragraph in p_tags:
            if (i < (len(p_tags)-3)) and (i >= 2):
                raw_lines.append(paragraph.text)
            i += 1
            
        logger.log_info(f"finished scrapping url='{url}'")
    except Exception as err:
        logger.log_error(err)  # type: ignore
        
    return raw_lines

def scrape_transcripts_from_website(raw_data_directory: str, dataset_name: str, force: bool = False) -> None:
    """scrapes raw page text from root url and saves to invidual text files inside the raw_data_directory path."""
    url_root = "https://transcripts.foreverdreaming.org/viewtopic.php?f=165&t="
    
    first_breaking_bad_pid = 44
    last_breaking_bad_pid = 107
    transcript_count = 0
    
     # TODO(Sean) there are 62 total breaking bad episodes, but we are downloading 64...
    for i in range(first_breaking_bad_pid, last_breaking_bad_pid + 1):
        if i not in [45,106]:
            p_id = 10000 + i
            filepath: str = f"{raw_data_directory}/{dataset_name}/{p_id}_transcript.txt"
            
            transcript_count += 1
            
            # skip already downloaded transcripts when not forcing the download
            if os.path.isfile(filepath) and not force:
                logger.log_debug(f"skipping webscraping for transcript, cached file '{filepath}' exists")
                continue
            
            raw_lines = scrape_page(url=f"{url_root}{p_id}")
            
            logger.log_info(f"writing raw transcript text file '{filepath}'")
            with open(filepath, "w") as f:
                for line in raw_lines:
                    line = line.strip()
                    if len(line) > 0:
                        # TODO(Sean) figure out how to correctly write 'weird' characters to file
                        try:
                            f.write(f"{line}\n")
                        except Exception as err:
                            logger.log_error(err)  # type: ignore
            logger.log_info(f"finished writing raw transcript text file '{filepath}'")
    logger.log_info(f"{transcript_count} total transcripts scraped or cached.")
        
    