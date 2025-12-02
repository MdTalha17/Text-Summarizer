import os
from textSummarizer.logging import logger
from textSummarizer.entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
    
    def validate_all_files_exist(self) -> bool:
        try:
            validation_status = None

            all_files = os.listdir(os.path.join("artifacts", "data_ingestion", "samsum_dataset"))

            # Check if all required files are present
            missing_files = [file for file in self.config.ALL_REQUIRED_FILES if file not in all_files]
            
            if missing_files:
                validation_status = False
                logger.info(f"Missing required files: {missing_files}")
            else:
                validation_status = True
                logger.info("All required files are present")
            
            # Write status to file
            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Validation Status: {validation_status}")
                        
            return validation_status
        
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            raise e