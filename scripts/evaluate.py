"""Evaluate fraud detection performance using section-aware embeddings"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from configs import FraudDetectionConfig
from src.detection import InvestigationEngine
from src.retrieval import QueryGuidedVectorStore
from src.data.dataset import load_test_data, load_validation_data, load_training_data

@dataclass
class EvaluationDetail:
    """Detailed tracking of a single evaluation"""
    unique_id: str
    timestamp: str
    query_image_path: str
    ground_truth_label: int
    predicted_label: int
    status: str
    confidence: float
    reasoning: str
    num_similar_cases: int
    similar_cases: List[Dict[str, Any]] = field(default_factory=list)
    encoding_success: bool = False
    num_patches: int = 0
    sections_encoded: List[str] = field(default_factory=list)
    search_results: List[tuple] = field(default_factory=list)
    error_message: str = ""
    execution_time: float = 0.0

class EvaluationLogger:
    """
    Comprehensive logging system for fraud detection evaluation
    Tracks every step of the evaluation process with detailed information
    """

    def __init__(self, output_dir: str = "./evaluation_logs"):
        """Initialize evaluation logger"""
        self.output_dir = output_dir
        self.evaluations: List[EvaluationDetail] = []
        self.start_time = datetime.now()

        os.makedirs(output_dir, exist_ok=True)
        
        # Create session log file
        self.session_log_path = os.path.join(
            output_dir, 
            f"evaluation_session_{self.start_time.strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        self._init_session_log()

    def _init_session_log(self):
        """Initialize session log file with header"""
        with open(self.session_log_path, 'w', encoding='utf-8') as f:
            f.write(f"{'='*80}\n")
            f.write(f"FRAUD DETECTION EVALUATION SESSION LOG\n")
            f.write(f"{'='*80}\n")
            f.write(f"Session Start: {self.start_time.isoformat()}\n")
            f.write(f"{'='*80}\n\n")

    def log_message(self, message: str, level: str = "INFO"):
        """
        Log a message to both console and session log file
        
        Args:
            message: Message to log
            level: Log Level (INFO, WARNING, ERROR)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}"
        
        print(log_line)
        
        with open(self.session_log_path, 'a', encoding='utf-8') as f:
            f.write(log_line + "\n")

    def log_evaluation_start(self, unique_id: str, query_path: str, ground_truth: int):
        """Log the start of an evaluation"""
        self.log_message(f"{'-'*40}")
        self.log_message(f"Starting evaluation for: {unique_id}")
        self.log_message(f"Query Image: {query_path}")
        self.log_message(f"Ground Truth Label: {ground_truth}")
        self.log_message(f"{'-'*40}")

    def log_encoding_result(self, unique_id: str, success: bool, num_patches: int = 0, sections: List[str] = None, error: str = ""):
        """Log encoding results"""
        if success:
            self.log_message(f"Encoding successful for {unique_id}")
            self.log_message(f"  - Total patches: {num_patches}")
            self.log_message(f"  - Sections: {', '.join(sections or [])}")
        else:
            self.log_message(f"Encoding failed for {unique_id}: {error}", "ERROR")

    def log_search_results(self, unique_id: str, results: List[tuple]):
        """Log search results"""
        self.log_message(f"Search Results: Retrieved {len(results)} similar cases for {unique_id}")
        for i, (case_id, section, score) in enumerate(results[:10]):
            self.log_message(f"  {i+1}. Case: {case_id}, Section: {section}, Score: {score:.4f}")
        if len(results) > 10:
            self.log_message(f"  ... and {len(results) - 10} more cases")

    def log_fraud_detection_result(self, unique_id: str, status: str, confidence: float, reasoning: str, num_similar: int):
        """Log fraud detection result"""
        self.log_message(f"--- Fraud Detection Result for {unique_id}:")
        self.log_message(f"Status: {status}")
        self.log_message(f"Confidence: {confidence:.4f}")
        self.log_message(f"Similar cases used: {num_similar}")
        self.log_message(f"Reasoning: {reasoning[:200]}..." if len(reasoning) > 200 else f"Reasoning: {reasoning}")

    def add_evaluation(self, evaluation: EvaluationDetail):
        """Add completed evaluation to tracking"""
        self.evaluations.append(evaluation)
        
        # Log summary
        match = "✓" if evaluation.predicted_label == evaluation.ground_truth_label else "✗"
        self.log_message(f"[{match}] Evaluation complete: {evaluation.unique_id}")
        self.log_message(f"Predicted: {evaluation.predicted_label}, Actual: {evaluation.ground_truth_label}")
        self.log_message(f"Execution time: {evaluation.execution_time:.2f}s")
        self.log_message(f"{'='*80}\n")

    def save_detailed_report(self, evaluation: EvaluationDetail) -> str:
        """
        Save detailed report for a single evaluation
        
        Args:
            evaluation: EvaluationDetail to save
            
        Returns:
            Path to saved report
        """
        filename = f"{evaluation.unique_id}_evaluation_detail.txt"
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"{'='*80}\n")
            f.write(f"DETAILED EVALUATION REPORT\n")
            f.write(f"Case ID: {evaluation.unique_id}\n")
            f.write(f"Timestamp: {evaluation.timestamp}\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"--- Input Information ---\n")
            f.write(f"{'='*80}\n")
            f.write(f"Query Image Path: {evaluation.query_image_path}\n")
            f.write(f"Ground Truth Label: {evaluation.ground_truth_label} ({'FRAUD' if evaluation.ground_truth_label == 1 else 'LEGITIMATE'})\n")
            f.write(f"{'='*80}\n\n")

            f.write(f"--- Encoding Phase ---\n")
            f.write(f"{'='*80}\n")
            f.write(f"Encoding Success: {evaluation.encoding_success}\n")
            if evaluation.encoding_success:
                f.write(f"Number of Patches Encoded: {evaluation.num_patches}\n")
                f.write(f"Sections Encoded: {', '.join(evaluation.sections_encoded)}\n")
            else:
                f.write(f"Error: {evaluation.error_message}\n")
            f.write(f"{'='*80}\n\n")
            
            if evaluation.encoding_success:
                f.write(f"--- Similar Case Retrieval ---\n")
                f.write(f"{'='*80}\n")
                f.write(f"Total Retrieved: {len(evaluation.search_results)}\n")
                f.write(f"Valid Similar Cases: {evaluation.num_similar_cases}\n")
                
                if evaluation.search_results:
                    f.write(f"Top Retrieved Cases:\n")
                    for i, (case_id, section, score) in enumerate(evaluation.search_results[:15]):
                        f.write(f"  {i+1}. Case ID: {case_id}\n")
                        f.write(f"     Section: {section}\n")
                        f.write(f"     Similarity Score: {score:.4f}\n")
                f.write(f"{'='*80}\n\n")
                
                f.write(f"--- Similar Case Images ---\n")
                f.write(f"{'='*80}\n")
                if evaluation.similar_cases:
                    for i, case_info in enumerate(evaluation.similar_cases):
                        f.write(f"  {i+1}. Path: {case_info.get('path', 'N/A')}\n")
                        f.write(f"     Similarity Score: {case_info.get('score', 0):.4f}\n")
                        f.write(f"     Section: {case_info.get('section', 'N/A')}\n")
                else:
                    f.write("No valid similar case images found.\n")
                f.write(f"{'='*80}\n\n")
            
            f.write(f"--- Fraud Detection Result ---\n")
            f.write(f"{'='*80}\n")
            f.write(f"Status: {evaluation.status}\n")
            f.write(f"Confidence: {evaluation.confidence:.4f}\n")
            f.write(f"Predicted Label: {evaluation.predicted_label} ({'FRAUD' if evaluation.predicted_label == 1 else 'LEGITIMATE'})\n")
            f.write(f"Ground Truth Label: {evaluation.ground_truth_label} ({'FRAUD' if evaluation.ground_truth_label == 1 else 'LEGITIMATE'})\n")
            f.write(f"Prediction Correct: {'YES' if evaluation.predicted_label == evaluation.ground_truth_label else 'NO'}\n\n")
            
            f.write(f"--- Reasoning ---\n")
            f.write(f"{'='*80}\n")
            f.write(f"{evaluation.reasoning}\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"--- Performance Metrics ---\n")
            f.write(f"{'='*80}\n")
            f.write(f"Execution Time: {evaluation.execution_time:.3f} seconds\n")
            f.write(f"{'='*80}\n")

        return output_path

    def save_session_summary(self, metrics: Dict[str, Any], test_size: int):
        """Save session summary with overall metrics"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        with open(self.session_log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"EVALUATION SESSION SUMMARY\n")
            f.write(f"{'='*80}\n")
            
            f.write(f"Session Duration: {duration:.2f} seconds\n")
            f.write(f"Test Set Size: {test_size}\n")
            f.write(f"Successfully Evaluated: {len(self.evaluations)}\n")
            f.write(f"Failed/Skipped: {test_size - len(self.evaluations)}\n\n")
            
            f.write(f"--- Performance Metrics ---\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1-Score: {metrics['f1_score']:.4f}\n")
            f.write(f"Avg Confidence: {metrics['avg_confidence']:.4f}\n\n")
            
            f.write(f"--- Confusion Matrix ---\n")
            f.write(f"{metrics['confusion_matrix']}\n\n")
            
            f.write(f"--- Classification Report ---\n")
            f.write(f"{metrics['classification_report']}\n")
            
            f.write(f"\n{'='*80}\n")
            f.write(f"Session End: {end_time.isoformat()}\n")
            f.write(f"{'='*80}\n")
            
        self.log_message(f"\n Session summary saved to: {self.session_log_path}")


class FraudDetectionEvaluator:
    """Evaluate fraud detection using section-aware embeddings"""
    
    def __init__(self, config: FraudDetectionConfig, encoder, vector_store):
        self.config = config
        self.data_dir = config.DATA_DIR
        self.encoder = encoder
        self.vector_store = vector_store
        self.fraud_detector = InvestigationEngine()
        self.logger = EvaluationLogger()
        
        stats = vector_store.get_stats()
        self.logger.log_message(f"Loaded vector store with {stats['total_patches']} patches")

    def evaluate(self, 
                 test_df: pd.DataFrame,
                 test_csv_path: str = "unknown",
                 k_similar: int = 5,
                 label_column: str = "SPI_label",
                 save_detailed_logs: bool = True) -> Dict:
        """
        Evaluate fraud detection using section-aware search
        
        Args:
            test_df: DataFrame with test data
            test_csv_path: Path to test CSV (for logging)
            k_similar: Number of similar cases to retrieve
            label_column: Column name for ground truth labels
            save_detailed_logs: Whether to save detailed logs for each evaluation
            
        Returns:
            Dict with evaluation metrics
        """
        self.logger.log_message(f"\nEvaluating data from: {test_csv_path}")
        # test_df is already passed in
        
        self.logger.log_message(f"Test set size: {len(test_df)} reports")
        label_dist = test_df[label_column].value_counts()
        self.logger.log_message(f"Label distribution:\n{label_dist}")
        
        predictions = []
        actuals = []
        confidences = []
        results_details = []
        
        self.logger.log_message("Starting section-aware fraud detection evaluation...")
        
        for pos, (idx, row) in enumerate(tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating")):
            eval_start_time = datetime.now()
            
            try:
                # Get ground truth label
                actual_label = row[label_column]
                unique_id = row['unique_ID']
                
                # Log evaluation start
                query_path = row.get('Path')
                if not query_path or pd.isna(query_path):
                    self.logger.log_message(f"Skipping {unique_id}: No image path", "WARNING")
                    continue
                    
                query_image_path = query_path if os.path.isabs(query_path) else os.path.join(self.data_dir, query_path)
                
                self.logger.log_evaluation_start(unique_id, query_image_path, actual_label)
                
                if not os.path.exists(query_image_path):
                    self.logger.log_message(f"Skipping {unique_id}: Image not found at {query_image_path}", "WARNING")
                    continue
                    
                # Create evaluation detail tracker
                evaluation = EvaluationDetail(
                    unique_id=unique_id,
                    timestamp=datetime.now().isoformat(),
                    query_image_path=query_image_path,
                    ground_truth_label=actual_label,
                    predicted_label=-1, # Will be updated
                    status="UNKNOWN",
                    confidence=0.0,
                    reasoning="",
                    num_similar_cases=0
                )
                
                # Encode query report
                self.logger.log_message(f"Encoding query report: {unique_id}")
                query_data = self.encoder.encode_report(query_image_path, report_id=unique_id)
                
                if not query_data:
                    evaluation.encoding_success = False
                    evaluation.error_message = "Failed to encode report"
                    self.logger.log_encoding_result(unique_id, False, error="Encoding returned None")
                    continue
                    
                # Create section-specific query embeddings
                query_emb_dict = {}
                sections_list = []
                total_patches = 0
                
                for section_name, section_data in query_data['sections'].items():
                    if 'patch_vectors' in section_data and len(section_data['patch_vectors']) > 0:
                        query_emb_dict[section_name] = np.array(section_data['patch_vectors'])
                        sections_list.append(section_name)
                        total_patches += len(section_data['patch_vectors'])
                        
                if not query_emb_dict:
                    evaluation.encoding_success = False
                    evaluation.error_message = "No patches generated"
                    self.logger.log_encoding_result(unique_id, False, error="No patches extracted")
                    continue
                    
                evaluation.encoding_success = True
                evaluation.num_patches = total_patches
                evaluation.sections_encoded = sections_list
                self.logger.log_encoding_result(unique_id, True, total_patches, sections_list)
                
                # Search for similar cases
                self.logger.log_message("Searching for similar cases...")
                similar_results = self.vector_store.search(
                    query_embedding=query_emb_dict,
                    sections=None,
                    top_k=15
                )
                
                evaluation.search_results = similar_results
                self.logger.log_search_results(unique_id, similar_results)
                
                if not similar_results:
                    self.logger.log_message("No similar results found. FAIL", "WARNING")
                    status = "FAIL"
                    confidence = 1.0
                    reasoning = "No similar cases found - report does not match any known legitimate patterns"
                    similar_case_image_paths = []
                else:
                    # Get similar case image paths
                    self.logger.log_message("Loading similar case images...")
                    train_df = load_training_data(self.config)
                    similar_case_image_paths = []
                    
                    for case_id, section, score in similar_results:
                        case_row = train_df[train_df['unique_ID'].astype(str) == str(case_id)]
                        if not case_row.empty:
                            case_path = case_row.iloc[0].get('Path')
                            if case_path and pd.notna(case_path):
                                full_case_path = case_path if os.path.isabs(case_path) else os.path.join(self.data_dir, case_path)
                                if os.path.exists(full_case_path):
                                    similar_case_image_paths.append({
                                        'case_id': case_id,
                                        'path': full_case_path,
                                        'score': score,
                                        'section': section
                                    })
                                    evaluation.similar_cases.append({
                                        'case_id': case_id,
                                        'path': full_case_path,
                                        'score': score,
                                        'section': section
                                    })
                    
                    if not similar_case_image_paths:
                        self.logger.log_message(f"Skipping {unique_id}: No valid similar case images found", "WARNING")
                        continue
                        
                    self.logger.log_message(f"Found {len(similar_case_image_paths)} valid similar case images")
                    
                    # Run fraud detection with IMAGES
                    self.logger.log_message("Running fraud detection...")
                    status, confidence, reasoning = self.fraud_detector.detect_fraud(
                        query_image_path=query_image_path,
                        similar_case_image_paths=similar_case_image_paths
                    )
                
                evaluation.num_similar_cases = len(similar_case_image_paths)
                evaluation.status = status
                evaluation.confidence = confidence
                evaluation.reasoning = reasoning
                
                self.logger.log_fraud_detection_result(unique_id, status, confidence, reasoning, len(similar_case_image_paths))
                
                # Convert to binary (FAIL=0, PASS=1, UNKNOWN=1)
                predicted = 1 if status in ["FAIL", "UNKNOWN"] else 0
                evaluation.predicted_label = predicted
                
                predictions.append(predicted)
                actuals.append(actual_label)
                confidences.append(confidence)
                
                # Calculate execution time
                eval_end_time = datetime.now()
                evaluation.execution_time = (eval_end_time - eval_start_time).total_seconds()
                
                # Add to tracking
                self.logger.add_evaluation(evaluation)
                
                # Save detailed report if requested
                if save_detailed_logs:
                    self.logger.save_detailed_report(evaluation)
                    
                results_details.append({
                    'unique_id': unique_id,
                    'predicted': predicted,
                    'confidence': confidence,
                    'status': status,
                    'reasoning': reasoning,
                    'num_similar': len(similar_case_image_paths),
                    'execution_time': evaluation.execution_time
                })
                
            except Exception as e:
                error_msg = f"Error processing {row.get('unique_ID', idx)}: {str(e)}"
                self.logger.log_message(error_msg, "ERROR")
                continue
                
        # Calculate metrics
        self.logger.log_message(f"\n{'='*80}")
        self.logger.log_message("EVALUATION RESULTS")
        self.logger.log_message(f"{'='*80}\n")
        
        metrics = self._calculate_metrics(actuals, predictions, confidences)
        
        # Log metrics
        self.logger.log_message(f"Total evaluated: {len(actuals)}")
        self.logger.log_message(f"Accuracy: {metrics['accuracy']:.4f}")
        self.logger.log_message(f"Precision: {metrics['precision']:.4f}")
        self.logger.log_message(f"Recall: {metrics['recall']:.4f}")
        self.logger.log_message(f"F1-Score: {metrics['f1_score']:.4f}")
        self.logger.log_message(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
        self.logger.log_message(f"Classification Report:\n{metrics['classification_report']}")
        self.logger.log_message(f"Avg Confidence: {metrics['avg_confidence']:.4f}")
        
        # Save session summary
        self.logger.save_session_summary(metrics, len(test_df))
        
        # Save detailed results CSV
        results_df = pd.DataFrame(results_details)
        output_path = test_csv_path.replace('.csv', '_fraud_detection_results.csv')
        results_df.to_csv(output_path, index=False)
        self.logger.log_message(f"\nDetailed results CSV saved to: {output_path}")
        
        metrics['results_details'] = results_details
        return metrics

    def _calculate_metrics(self, actuals, predictions, confidences):
        """Calculate evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(actuals, predictions),
            'precision': precision_score(actuals, predictions, zero_division=0),
            'recall': recall_score(actuals, predictions, zero_division=0),
            'f1_score': f1_score(actuals, predictions, zero_division=0),
            'confusion_matrix': confusion_matrix(actuals, predictions),
            'classification_report': classification_report(actuals, predictions, zero_division=0),
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'num_evaluated': len(actuals)
        }
        return metrics

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate fraud detection performance")
    
    parser.add_argument("--model", type=str, default="jina", choices=["jina", "colqwen"],
                        help="Model type to use for encoding (default: jina)")
    parser.add_argument("--eval-type", type=str, default="test", choices=["test", "validation"],
                        help="Evaluation set to use (default: test)")
    parser.add_argument("--k-similar", type=int, default=15,
                        help="Number of similar cases to retrieve (default: 15)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Top K results per section (default: 5)")
    parser.add_argument("--vector-store-path", type=str, default=None,
                        help="Path to vector store (default: from config)")
    parser.add_argument("--no-detailed-logs", action="store_true",
                        help="Disable detailed per-case logs")
                        
    args = parser.parse_args()
    
    # Initialize config
    config = FraudDetectionConfig()
    
    # Set vector store path based on model (matches inference.py behavior)
    if args.vector_store_path:
        config.VECTOR_STORE_PATH = args.vector_store_path
    else:
        config.VECTOR_STORE_PATH = f"./vectorstore_{args.model}"
        
    # Set model based on argument
    if args.model == "colqwen":
        config.MODEL_NAME = config.COLQWEN_MODEL_NAME
        
    print(f"Initializing evaluation with {args.model} model...")
    print(f"Evaluation type: {args.eval_type}")
    print(f"K similar: {args.k_similar}, Top K per section: {args.top_k}")
    
    # Import encoder here to avoid loading models if just checking help
    from src.models import ReceiptMultiModalEncoder
    
    # Initialize components
    print("Loading encoder...")
    encoder = ReceiptMultiModalEncoder(config)
    
    print("Loading vector store...")
    vector_store = QueryGuidedVectorStore(config)
    
    # Get test data
    if args.eval_type == "test":
        test_csv_path = config.TEST_DATA_PATH
        test_df = load_test_data(config)
    else:
        test_csv_path = config.VALIDATION_DATA_PATH
        test_df = load_validation_data(config)
        
    print(f"Using evaluation data: {test_csv_path}")
    
    # Initialize evaluator and run
    evaluator = FraudDetectionEvaluator(config, encoder, vector_store)
    
    metrics = evaluator.evaluate(
        test_df=test_df,
        test_csv_path=test_csv_path,
        k_similar=args.k_similar,
        save_detailed_logs=not args.no_detailed_logs
    )
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print("="*80)
    
    return metrics

if __name__ == "__main__":
    main()

