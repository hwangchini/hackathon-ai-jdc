import sys
import os
from datetime import datetime
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from scrapers.drugs_scraper import DrugsScraper
from processors.llm_processor import LLMProcessor
from data_manager import DataManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def scrape_and_update_medicine(medicine_name: str, data_manager: DataManager, 
                                scraper: DrugsScraper, processor: LLMProcessor):
    """Scrape và cập nhật thông tin cho một loại thuốc"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {medicine_name}")
    logger.info(f"{'='*60}")
    
    # 1. Scrape data
    logger.info("Step 1: Scraping data from drugs.com...")
    raw_data = scraper.scrape_drug_info(medicine_name)
    
    if not raw_data:
        logger.warning(f"Could not scrape data for {medicine_name}, trying LLM only...")
        # Fallback: Dùng LLM với thông tin hiện có
        existing_data = data_manager.get_medicine(medicine_name)
        processed_data = processor.translate_and_enrich(medicine_name, existing_data)
    else:
        # 2. Process with LLM
        logger.info("Step 2: Processing with LLM (translating & normalizing)...")
        sample_structure = data_manager.get_sample_structure()
        processed_data = processor.process_medicine_data(raw_data, sample_structure)
    
    if processed_data:
        # 3. Update database
        logger.info("Step 3: Updating database...")
        processed_data['last_updated'] = datetime.now().strftime('%Y-%m-%d')
        data_manager.add_medicine(processed_data)
        logger.info(f"✓ Successfully updated {medicine_name}")
        return True
    else:
        logger.error(f"✗ Failed to process {medicine_name}")
        return False


def main():
    """Main function"""
    # Initialize
    json_path = os.path.join(os.path.dirname(__file__), 'data', 'documents', 'medicines.json')
    data_manager = DataManager(json_path)
    scraper = DrugsScraper()
    processor = LLMProcessor()
    
    print("\n" + "="*60)
    print("MEDICINE DATA SCRAPER & UPDATER")
    print("="*60)
    
    # Menu
    print("\nOptions:")
    print("1. Update incomplete medicines")
    print("2. Add new medicine")
    print("3. Update specific medicine")
    print("4. Update all medicines")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        # Update incomplete medicines
        incomplete = data_manager.get_incomplete_medicines()
        print(f"\nFound {len(incomplete)} incomplete medicines:")
        for med in incomplete:
            print(f"  - {med.get('medicine_name')}")
        
        if incomplete:
            confirm = input("\nProceed with update? (y/n): ").strip().lower()
            if confirm == 'y':
                success_count = 0
                for med in incomplete:
                    if scrape_and_update_medicine(
                        med.get('medicine_name'), 
                        data_manager, 
                        scraper, 
                        processor
                    ):
                        success_count += 1
                
                data_manager.save_data()
                print(f"\n✓ Updated {success_count}/{len(incomplete)} medicines")
    
    elif choice == '2':
        # Add new medicine
        medicine_name = input("\nEnter medicine name: ").strip()
        if medicine_name:
            if scrape_and_update_medicine(medicine_name, data_manager, scraper, processor):
                data_manager.save_data()
                print(f"\n✓ Successfully added {medicine_name}")
    
    elif choice == '3':
        # Update specific medicine
        all_medicines = data_manager.get_all_medicines()
        print("\nCurrent medicines:")
        for i, med in enumerate(all_medicines, 1):
            print(f"{i}. {med.get('medicine_name')}")
        
        try:
            idx = int(input("\nEnter medicine number: ")) - 1
            if 0 <= idx < len(all_medicines):
                medicine_name = all_medicines[idx].get('medicine_name')
                if scrape_and_update_medicine(medicine_name, data_manager, scraper, processor):
                    data_manager.save_data()
                    print(f"\n✓ Successfully updated {medicine_name}")
        except ValueError:
            print("Invalid input")
    
    elif choice == '4':
        # Update all
        all_medicines = data_manager.get_all_medicines()
        print(f"\nWill update all {len(all_medicines)} medicines")
        confirm = input("This may take a while. Continue? (y/n): ").strip().lower()
        
        if confirm == 'y':
            success_count = 0
            for med in all_medicines:
                if scrape_and_update_medicine(
                    med.get('medicine_name'), 
                    data_manager, 
                    scraper, 
                    processor
                ):
                    success_count += 1
            
            data_manager.save_data()
            print(f"\n✓ Updated {success_count}/{len(all_medicines)} medicines")
    
    else:
        print("Invalid choice")
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
