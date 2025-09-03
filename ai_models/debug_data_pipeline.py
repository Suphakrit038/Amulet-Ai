"""
Debug script to test data pipeline creation
"""
import logging
from pathlib import Path
from advanced_data_pipeline import AdvancedDataPipeline, DataPipelineConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_pipeline():
    """Test basic data pipeline creation"""
    logger.info("🔧 Testing data pipeline creation...")
    
    # Create minimal config
    config = DataPipelineConfig(
        split_path="dataset_split",
        batch_size=2,
        num_workers=0,  # No multiprocessing
        quality_threshold=0.3
    )
    
    logger.info("📊 Creating pipeline...")
    pipeline = AdvancedDataPipeline(config)
    
    logger.info("📂 Creating datasets...")
    try:
        datasets = pipeline.create_datasets()
        logger.info(f"✅ Created datasets: {list(datasets.keys())}")
        
        for split_name, dataset in datasets.items():
            logger.info(f"📊 {split_name}: {len(dataset)} images")
            
    except Exception as e:
        logger.error(f"❌ Failed to create datasets: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    
    logger.info("📊 Creating dataloaders...")
    try:
        dataloaders = pipeline.create_dataloaders()
        logger.info(f"✅ Created dataloaders: {list(dataloaders.keys())}")
        
        # Test loading one batch
        for split_name, dataloader in dataloaders.items():
            logger.info(f"🔍 Testing {split_name} dataloader...")
            try:
                batch = next(iter(dataloader))
                logger.info(f"✅ {split_name} batch loaded successfully")
                break  # Just test one
            except Exception as e:
                logger.error(f"❌ Failed to load {split_name} batch: {e}")
                
    except Exception as e:
        logger.error(f"❌ Failed to create dataloaders: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    
    logger.info("🎉 Data pipeline test completed!")
    return True

if __name__ == "__main__":
    test_data_pipeline()
