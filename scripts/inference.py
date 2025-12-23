#!/usr/bin/env python
import argparse
import os
import sys
import numpy as np
sys.path.insert(0, '.')

from configs import FraudDetectionConfig
from src.models import ReceiptMultiModalEncoder
from src.retrieval import QueryGuidedVectorStore
from src.data.dataset import load_test_data, get_image_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='jina', choices=['jina', 'colqwen'])
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--top-k', type=int, default=None, help="Total results to return")
    parser.add_argument('--k-per-section', type=int, default=None, help="Results per section before aggregation")
    parser.add_argument('--sections', type=str, nargs='+', default=None)
    parser.add_argument('--vectorstore-path', type=str, default=None)
    args = parser.parse_args()

    config = FraudDetectionConfig()

    # Override config with CLI args
    if args.model == 'colqwen':
        config.MODEL_NAME = config.COLQWEN_MODEL_NAME
    if args.vectorstore_path:
        config.VECTOR_STORE_PATH = args.vectorstore_path
    else:
        config.VECTOR_STORE_PATH = f"./vectorstore_{args.model}"
    
    top_k = args.top_k or config.DEFAULT_TOP_K
    k_per_section = args.k_per_section or config.DEFAULT_TOP_K_PER_SECTION
    
    encoder = ReceiptMultiModalEncoder(config=config)
    vector_store = QueryGuidedVectorStore(config=config)
    
    test_df = load_test_data(config)
    
    if args.index >= len(test_df):
        raise ValueError(f"Index {args.index} out of range (max: {len(test_df)-1})")
        
    selected_row = test_df.iloc[args.index]
    report_id = selected_row['unique_ID']
    print(f"Query Report: Index {args.index} | ID: {report_id}")
    
    full_path = get_image_path(selected_row, config.DATA_DIR)
    if not full_path or not os.path.exists(full_path):
        raise FileNotFoundError(f"Image not found: {full_path}")
        
    query_data = encoder.encode_report(full_path, report_id=report_id)
    if not query_data:
        raise RuntimeError("Failed to encode query report")
        
    query_emb_dict = {}
    for section_name, section_data in query_data['sections'].items():
        if 'patch_vectors' in section_data and len(section_data['patch_vectors']) > 0:
            query_emb_dict[section_name] = np.array(section_data['patch_vectors'])
            
    if not query_emb_dict:
        raise RuntimeError("No patches found in query report")
        
    results = vector_store.search(
        query_embedding=query_emb_dict,
        sections=args.sections,
        top_k=top_k,
        top_k_per_section=k_per_section
    )
    
    print(f"\nTop {len(results)} Similar Reports:\n")
    for i, (case_id, section, score) in enumerate(results, 1):
        print(f" {i}. {case_id:<15} | Section: {section:<10} | Score: {score:.3f}")
        
    return results
    
if __name__ == '__main__':
    main()

