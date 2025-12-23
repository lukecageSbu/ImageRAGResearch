#!/usr/bin/env python
import argparse
import sys
sys.path.insert(0, '.')

from configs import FraudDetectionConfig
from src.models import ReceiptMultiModalEncoder, process_report_dataset
from src.retrieval import QueryGuidedVectorStore
from src.data.dataset import load_training_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='jina', choices=['jina', 'colqwen'])
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--vectorstore-path', type=str, default=None)
    args = parser.parse_args()

    config = FraudDetectionConfig()

    # Override config with CLI args
    if args.model == 'colqwen':
        config.MODEL_NAME = config.COLQWEN_MODEL_NAME
    if args.threshold:
        config.SIMILARITY_THRESHOLD = args.threshold
    if args.device:
        config.DEVICE = args.device
    if args.vectorstore_path:
        config.VECTOR_STORE_PATH = args.vectorstore_path
    else:
        config.VECTOR_STORE_PATH = f"./vectorstore_{args.model}"

    print(f"Training with model: {config.MODEL_NAME}")
    print(f"Vector store: {config.VECTOR_STORE_PATH}")

    encoder = ReceiptMultiModalEncoder(config=config)
    vector_store = QueryGuidedVectorStore(config=config)
    
    train_df = load_training_data(config)
    print(f"Loaded {len(train_df)} training reports")
    
    encoded_reports = process_report_dataset(
        encoder=encoder,
        df=train_df,
        data_dir=config.DATA_DIR
    )
    
    if encoded_reports:
        vector_store.add_reports(encoded_reports)
        stats = vector_store.get_stats()
        print(f"Training complete - {stats['total_patches']} patches stored")
        return True
    else:
        print("Training failed - no reports encoded")
        return False

if __name__ == '__main__':
    main()

