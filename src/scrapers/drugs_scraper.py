import requests
from bs4 import BeautifulSoup
import time
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrugsScraper:
    """Scraper for drugs.com website"""
    
    BASE_URL = "https://www.drugs.com"
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
    
    def search_drug(self, drug_name: str) -> Optional[str]:
        """Tìm kiếm thuốc và trả về URL chi tiết"""
        try:
            search_url = f"{self.BASE_URL}/{drug_name.lower().replace(' ', '-')}.html"
            response = self.session.get(search_url, timeout=10)
            
            if response.status_code == 200:
                return search_url
            
            # Fallback: search
            search_url = f"{self.BASE_URL}/search.php?searchterm={drug_name}"
            response = self.session.get(search_url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            first_result = soup.select_one('.ddc-search-result a')
            if first_result:
                return self.BASE_URL + first_result['href']
                
            return None
        except Exception as e:
            logger.error(f"Error searching drug {drug_name}: {e}")
            return None
    
    def scrape_drug_info(self, drug_name: str) -> Optional[Dict]:
        """Scrape thông tin chi tiết về thuốc"""
        try:
            drug_url = self.search_drug(drug_name)
            if not drug_url:
                logger.warning(f"Could not find URL for {drug_name}")
                return None
            
            logger.info(f"Scraping {drug_name} from {drug_url}")
            response = self.session.get(drug_url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            data = {
                'medicine_name': drug_name,
                'generic_name': self._extract_generic_name(soup),
                'brand_names': self._extract_brand_names(soup),
                'indications': self._extract_indications(soup),
                'dosage': self._extract_dosage(soup),
                'contraindications': self._extract_contraindications(soup),
                'side_effects': self._extract_side_effects(soup),
                'warnings': self._extract_warnings(soup),
                'source': 'Drugs.com',
                'reference_url': drug_url
            }
            
            time.sleep(1)  # Rate limiting
            return data
            
        except Exception as e:
            logger.error(f"Error scraping {drug_name}: {e}")
            return None
    
    def _extract_generic_name(self, soup: BeautifulSoup) -> str:
        """Extract generic name"""
        try:
            generic = soup.select_one('.contentBox h1')
            if generic:
                text = generic.get_text().strip()
                # Remove brand name in parentheses
                if '(' in text:
                    return text.split('(')[0].strip()
                return text
        except:
            pass
        return ""
    
    def _extract_brand_names(self, soup: BeautifulSoup) -> list:
        """Extract brand names"""
        brands = []
        try:
            brand_section = soup.find('p', string=lambda x: x and 'Brand names:' in x)
            if brand_section:
                text = brand_section.get_text()
                brands_text = text.split('Brand names:')[1].strip()
                brands = [b.strip() for b in brands_text.split(',')]
        except:
            pass
        return brands
    
    def _extract_indications(self, soup: BeautifulSoup) -> list:
        """Extract indications/uses"""
        indications = []
        try:
            # Look for "Uses" section
            uses_heading = soup.find(['h2', 'h3'], string=lambda x: x and 'uses' in x.lower())
            if uses_heading:
                content = uses_heading.find_next(['p', 'ul'])
                if content:
                    if content.name == 'ul':
                        indications = [li.get_text().strip() for li in content.find_all('li')]
                    else:
                        text = content.get_text().strip()
                        indications = [s.strip() for s in text.split('.') if s.strip()]
        except:
            pass
        return indications[:10]  # Limit to 10 items
    
    def _extract_dosage(self, soup: BeautifulSoup) -> Dict:
        """Extract dosage information"""
        dosage = {}
        try:
            dosage_heading = soup.find(['h2', 'h3'], string=lambda x: x and 'dosage' in x.lower())
            if dosage_heading:
                content = dosage_heading.find_next(['p', 'div'])
                if content:
                    text = content.get_text().strip()
                    dosage['adult'] = text[:200]  # Limit length
        except:
            pass
        return dosage
    
    def _extract_contraindications(self, soup: BeautifulSoup) -> list:
        """Extract contraindications"""
        contraindications = []
        try:
            section = soup.find(['h2', 'h3'], string=lambda x: x and 'contraindication' in x.lower())
            if section:
                content = section.find_next(['p', 'ul'])
                if content:
                    if content.name == 'ul':
                        contraindications = [li.get_text().strip() for li in content.find_all('li')]
                    else:
                        text = content.get_text().strip()
                        contraindications = [s.strip() for s in text.split('.') if s.strip()]
        except:
            pass
        return contraindications[:10]
    
    def _extract_side_effects(self, soup: BeautifulSoup) -> list:
        """Extract side effects"""
        side_effects = []
        try:
            section = soup.find(['h2', 'h3'], string=lambda x: x and 'side effect' in x.lower())
            if section:
                content = section.find_next(['p', 'ul'])
                if content:
                    if content.name == 'ul':
                        side_effects = [li.get_text().strip() for li in content.find_all('li')]
                    else:
                        text = content.get_text().strip()
                        side_effects = [s.strip() for s in text.split('.') if s.strip()]
        except:
            pass
        return side_effects[:15]
    
    def _extract_warnings(self, soup: BeautifulSoup) -> str:
        """Extract warnings"""
        try:
            section = soup.find(['h2', 'h3'], string=lambda x: x and 'warning' in x.lower())
            if section:
                content = section.find_next('p')
                if content:
                    return content.get_text().strip()[:300]
        except:
            pass
        return ""
