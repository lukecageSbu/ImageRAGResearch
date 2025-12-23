import torch
import numpy as np
import pandas as pd
from transformers import AutoModel
from typing import List, Dict, Optional
import os
from tqdm import tqdm

from configs import FraudDetectionConfig

class ReceiptMultiModalEncoder:
    
    SECTIONS = {
        'header': """merchant information including: store name, company logo, merchant address, 
phone number, company registration number (SSM/GST/VAT ID), receipt title (e.g., TAX INVOICE, CASH BILL)""",

        'items': """purchased items list including: product descriptions, quantity, unit prices, 
line item totals, product codes or SKUs""",

        'payment': """payment and total details including: subtotal, tax amounts (GST/SST/VAT), 
service charge, rounding adjustment, grand total, payment method (Cash, Credit Card, E-wallet), 
change amount""",

        'metadata': """transaction metadata including: date of purchase, time of purchase, 
receipt/invoice number, cashier name or ID, register/terminal number, member/loyalty card details""",

        'footer': """receipt footer information including: thank you message, return/exchange policy, 
social media links, barcodes, QR codes, website URL, end of receipt markers"""
    }

    def __init__(self, config: FraudDetectionConfig = None):
        self.config = config or FraudDetectionConfig()
        self.device = self.config.DEVICE
        self.similarity_threshold = self.config.SIMILARITY_THRESHOLD
        self.model_name = self.config.MODEL_NAME
        
        if "colqwen" in self.model_name.lower():
            self.model_type = "colqwen"
            self._load_colqwen()
        else:
            self.model_type = "jina"
            self._load_jina()
            
        self._encode_section_fields()

    def _load_jina(self):
        self.jina_device = self.config.get_device("jina")
        print(f"Loading Jina v4 on {self.jina_device}...")
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model.to(self.jina_device)
        self.model.eval()
        self.processor = None
        self.embedding_dim = self.config.EMBEDDING_DIM

    def _load_colqwen(self):
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
        
        self.colqwen_device = self.config.get_device("colqwen")
        print(f"Loading ColQwen2.5 on {self.colqwen_device}...")
        
        self.model = ColQwen2_5.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16
        ).eval()
        
        self.model.to(self.colqwen_device)
        self.processor = ColQwen2_5_Processor.from_pretrained(self.model_name)
        self.embedding_dim = self.config.EMBEDDING_DIM

    def _encode_section_fields(self):
        print("Encoding section fields...")
        section_descriptions = list(self.SECTIONS.values())
        
        with torch.no_grad():
            if self.model_type == "jina":
                field_embeddings = self.model.encode_text(
                    texts=section_descriptions,
                    task="retrieval",
                    prompt_name="query",
                    return_multivector=True
                )
            else:
                batch_queries = self.processor.process_queries(section_descriptions).to(self.colqwen_device)
                field_embeddings = self.model(**batch_queries)
                
        self.section_embeddings = {}
        for section_name, emb in zip(self.SECTIONS.keys(), field_embeddings):
            if torch.is_tensor(emb):
                if emb.dtype == torch.bfloat16:
                    emb = emb.float()
                emb = emb.cpu().numpy()
            self.section_embeddings[section_name] = emb

    def encode_report(self, image_path: str, report_id: str = None) -> Optional[Dict]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        with torch.no_grad():
            if self.model_type == "jina":
                image_multivec = self.model.encode_image(
                    images=[image_path],
                    task="retrieval",
                    return_multivector=True
                )[0]
            else:
                from PIL import Image
                image = Image.open(image_path)
                batch_images = self.processor.process_images([image]).to(self.colqwen_device)
                image_multivec = self.model(**batch_images)[0]
                
        if torch.is_tensor(image_multivec):
            if image_multivec.dtype == torch.bfloat16:
                image_multivec = image_multivec.float()
            image_multivec = image_multivec.cpu().numpy()
            
        report_data = {
            'report_id': report_id or os.path.basename(image_path),
            'image_path': image_path,
            'total_patches': len(image_multivec),
            'sections': {}
        }
        
        total_stored = 0
        for section_name, section_emb in self.section_embeddings.items():
            relevant_patches = self._filter_patches_for_section(image_multivec, section_emb)
            if relevant_patches:
                report_data['sections'][section_name] = relevant_patches
                total_stored += len(relevant_patches['patch_indices'])
                
        report_data['stored_patches'] = total_stored
        report_data['storage_ratio'] = total_stored / (len(image_multivec) * len(self.SECTIONS))
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        return report_data

    def _filter_patches_for_section(self, image_patches: np.ndarray, section_embedding: np.ndarray) -> Optional[Dict]:
        relevant_indices = []
        similarities = []
        
        for img_idx, img_patch in enumerate(image_patches):
            max_sim = 0
            for section_vec in section_embedding:
                sim = np.dot(img_patch, section_vec) / (
                    np.linalg.norm(img_patch) * np.linalg.norm(section_vec) + 1e-8
                )
                max_sim = max(max_sim, sim)
                
            if max_sim >= self.similarity_threshold:
                relevant_indices.append(img_idx)
                similarities.append(float(max_sim))
                
        if not relevant_indices:
            return None
            
        return {
            'patch_indices': relevant_indices,
            'patch_vectors': image_patches[relevant_indices],
            'similarities': similarities
        }

def process_report_dataset(encoder: ReceiptMultiModalEncoder, 
                         df: pd.DataFrame, 
                         data_dir: str = "./data") -> List[Dict]:
    
    import gc
    
    encoded_reports = []
    skipped = 0
    errors = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Encoding reports"):
        if 'Path' not in row or pd.isna(row['Path']):
            skipped += 1
            continue
            
        image_path = str(row['Path'])
        if not os.path.isabs(image_path):
            image_path = os.path.join(data_dir, image_path)
            
        report_id = str(row.get('unique_ID', f"report_{idx}"))
        
        try:
            encoded_report = encoder.encode_report(image_path, report_id)
            if encoded_report:
                encoded_reports.append(encoded_report)
        except torch.cuda.OutOfMemoryError:
            print(f"\n[OOM] Skipping {report_id} - CUDA out of memory")
            errors.append((report_id, "OOM"))
            torch.cuda.empty_cache()
        except FileNotFoundError as e:
            print(f"\n[SKIP] Skipping {report_id} - File not found")
            errors.append((report_id, "FileNotFound"))
        except Exception as e:
            print(f"\n[ERROR] Skipping {report_id} - {type(e).__name__}: {str(e)[:100]}")
            errors.append((report_id, str(type(e).__name__)))
            
        gc.collect()
        if encoder.device == "cuda":
            torch.cuda.empty_cache()
            
    print(f"\nSuccessfully encoded: {len(encoded_reports)} reports")
    print(f"Skipped (no path): {skipped}")
    print(f"Errors: {len(errors)}")
    
    if encoded_reports:
        avg_storage = np.mean([r['storage_ratio'] for r in encoded_reports])
        avg_patches = np.mean([r['stored_patches'] for r in encoded_reports])
        print(f"Storage efficiency: {avg_storage*100:.1f}%")
        print(f"Average patches per report: {avg_patches:.0f}")
        
    return encoded_reports

