import json
import os
from typing import Dict, List
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    """Manage medicine data in JSON file"""
    
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.data = self._load_data()
    
    def _load_data(self) -> Dict:
        """Load data from JSON file"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"File not found: {self.json_path}")
            return {"medicines": []}
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
            return {"medicines": []}
    
    def save_data(self):
        """Save data to JSON file with backup"""
        try:
            # Create backup
            if os.path.exists(self.json_path):
                backup_path = f"{self.json_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    backup_data = f.read()
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(backup_data)
                logger.info(f"Backup created: {backup_path}")
            
            # Save new data
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            logger.info(f"Data saved to {self.json_path}")
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def get_medicine(self, medicine_name: str) -> Dict:
        """Get medicine by name"""
        for med in self.data.get('medicines', []):
            if med.get('medicine_name', '').lower() == medicine_name.lower():
                return med
        return None
    
    def add_medicine(self, medicine_data: Dict):
        """Add new medicine or update existing"""
        medicine_name = medicine_data.get('medicine_name')
        
        # Check if medicine exists
        for i, med in enumerate(self.data['medicines']):
            if med.get('medicine_name', '').lower() == medicine_name.lower():
                # Update existing
                self.data['medicines'][i] = medicine_data
                logger.info(f"Updated medicine: {medicine_name}")
                return
        
        # Add new
        self.data['medicines'].append(medicine_data)
        logger.info(f"Added new medicine: {medicine_name}")
    
    def get_all_medicines(self) -> List[Dict]:
        """Get all medicines"""
        return self.data.get('medicines', [])
    
    def get_incomplete_medicines(self) -> List[Dict]:
        """Get medicines with incomplete data"""
        incomplete = []
        required_fields = [
            'medicine_name', 'generic_name', 'category', 
            'indications', 'dosage', 'contraindications', 
            'side_effects', 'warnings'
        ]
        
        for med in self.data.get('medicines', []):
            is_incomplete = False
            for field in required_fields:
                value = med.get(field)
                if not value or (isinstance(value, (list, dict, str)) and not value):
                    is_incomplete = True
                    break
            
            if is_incomplete:
                incomplete.append(med)
        
        return incomplete
    
    def get_sample_structure(self) -> Dict:
        """Get sample medicine structure from existing data"""
        medicines = self.data.get('medicines', [])
        if medicines:
            return medicines[0]
        return {}
