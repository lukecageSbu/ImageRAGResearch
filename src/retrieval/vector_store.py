import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from configs import FraudDetectionConfig

try:
    import lancedb
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False

class QueryGuidedVectorStore:
    
    SECTIONS = ['header', 'items', 'payment', 'metadata', 'footer']
    
    def __init__(self, config: FraudDetectionConfig = None):
        if not LANCEDB_AVAILABLE:
            raise RuntimeError("Install lancedb: pip install lancedb")
            
        self.config = config or FraudDetectionConfig()
        self.db_path = self.config.VECTOR_STORE_PATH
        self.max_patches = self.config.MAX_PATCHES
        self.patch_dim = None # Auto-detected from data
        
        os.makedirs(self.db_path, exist_ok=True)
        
        self.db = lancedb.connect(self.db_path)
        self.table_name = "receipts"
        
        if self.table_name in self.db.table_names():
            self.table = self.db.open_table(self.table_name)
            self._detect_patch_dim_from_table()
        else:
            self.table = None
            
    def _detect_patch_dim_from_table(self):
        """Auto-detect patch_dim from existing table data."""
        try:
            df = self.table.to_pandas()
            if len(df) > 0:
                first_embedding = df[f'{self.SECTIONS[0]}_embedding'].iloc[0]
                self.patch_dim = len(first_embedding) // self.max_patches
                print(f"Auto-detected patch_dim={self.patch_dim} from table")
        except Exception as e:
            print(f"Could not auto-detect patch dimension: {e}")

    def _flatten_section(self, patches: List[np.ndarray]) -> tuple:
        patch_list = []
        for p in patches:
            p = np.asarray(p, dtype=np.float32)
            
            # Auto-detect patch_dim from first valid patch
            if self.patch_dim is None:
                self.patch_dim = len(p)
                print(f"Auto-detected patch_dim={self.patch_dim} from data")
                
            if len(p) == self.patch_dim:
                patch_list.append(p)
                
        if len(patch_list) == 0:
            if self.patch_dim is None:
                self.patch_dim = 128 # Fallback default
            flattened = np.zeros(self.max_patches * self.patch_dim, dtype=np.float32)
            return flattened.tolist(), 0
            
        patch_array = np.stack(patch_list)
        n_patches = len(patch_list)
        
        if n_patches < self.max_patches:
            padding = np.zeros((self.max_patches - n_patches, self.patch_dim), dtype=np.float32)
            patch_array = np.vstack([patch_array, padding])
        elif n_patches > self.max_patches:
            patch_array = patch_array[:self.max_patches]
            n_patches = self.max_patches
            
        return patch_array.flatten().tolist(), n_patches

    def _unflatten_section(self, flattened: List[float], n_patches: int) -> np.ndarray:
        if n_patches == 0:
            return np.zeros((0, self.patch_dim), dtype=np.float32)
            
        arr = np.array(flattened, dtype=np.float32)
        reshaped = arr.reshape(self.max_patches, self.patch_dim)
        return reshaped[:n_patches]

    def add_reports(self, encoded_reports: List[Dict]) -> bool:
        print(f"Adding {len(encoded_reports)} reports to vector store...")
        
        records = []
        for report in encoded_reports:
            record = {'report_id': report['report_id']}
            
            for section in self.SECTIONS:
                if section in report['sections']:
                    patches = report['sections'][section]['patch_vectors']
                    flattened, n_patches = self._flatten_section(patches)
                else:
                    flattened, n_patches = self._flatten_section([])
                    
                record[f'{section}_embedding'] = flattened
                record[f'{section}_n_patches'] = n_patches
                
            records.append(record)
            
        df = pd.DataFrame(records)
        
        if self.table is None:
            self.table = self.db.create_table(self.table_name, df, mode="overwrite")
        else:
            self.table.add(df)
            
        return True

    def _compute_maxsim(self, query_patches: np.ndarray, doc_patches: np.ndarray) -> float:
        if query_patches is None or doc_patches is None:
            return 0.0
            
        if len(query_patches) == 0 or len(doc_patches) == 0:
            return 0.0
            
        query_patches = np.asarray(query_patches, dtype=np.float32)
        doc_patches = np.asarray(doc_patches, dtype=np.float32)
        
        if len(query_patches.shape) == 1:
            query_patches = query_patches.reshape(1, -1)
        if len(doc_patches.shape) == 1:
            doc_patches = doc_patches.reshape(1, -1)
            
        query_norms = np.linalg.norm(query_patches, axis=1, keepdims=True) + 1e-8
        doc_norms = np.linalg.norm(doc_patches, axis=1, keepdims=True) + 1e-8
        
        query_norm = query_patches / query_norms
        doc_norm = doc_patches / doc_norms
        
        sim_matrix = np.matmul(query_norm, doc_norm.T)
        max_sims = np.max(sim_matrix, axis=1)
        return float(np.mean(max_sims))

    def search(self, 
               query_embedding, 
               sections: Optional[List[str]] = None,
               top_k: int = None,
               top_k_per_section: int = None) -> List[tuple]:
        if self.table is None:
            return []
            
        top_k = top_k or self.config.DEFAULT_TOP_K
        top_k_per_section = top_k_per_section or self.config.DEFAULT_TOP_K_PER_SECTION
        
        is_section_specific = isinstance(query_embedding, dict)
        
        if not is_section_specific:
            query_embedding = np.asarray(query_embedding, dtype=np.float32)
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
            query_embedding = {section: query_embedding for section in self.SECTIONS}
            
        sections_to_search = sections or self.SECTIONS
        
        # Validate section names
        invalid_sections = set(sections_to_search) - set(self.SECTIONS)
        if invalid_sections:
            raise ValueError(f"Invalid sections: {invalid_sections}. Valid: {self.SECTIONS}")
            
        all_results = []
        
        df = self.table.to_pandas()
        
        for section in sections_to_search:
            if section not in query_embedding or len(query_embedding[section]) == 0:
                continue
                
            embed_col = f'{section}_embedding'
            n_patches_col = f'{section}_n_patches'
            
            section_query = np.asarray(query_embedding[section], dtype=np.float32)
            if len(section_query.shape) == 1:
                section_query = section_query.reshape(1, -1)
                
            scores = []
            for _, row in df.iterrows():
                doc_patches = self._unflatten_section(row[embed_col], row[n_patches_col])
                maxsim_score = self._compute_maxsim(section_query, doc_patches)
                
                if maxsim_score > 0:
                    scores.append((row['report_id'], section, maxsim_score))
                    
            scores.sort(key=lambda x: x[2], reverse=True)
            all_results.extend(scores[:top_k_per_section])
            
        all_results.sort(key=lambda x: x[2], reverse=True)
        return all_results[:top_k]

    def get_stats(self) -> Dict:
        if self.table is None:
            return {'total_reports': 0, 'total_patches': 0, 'sections': {}}
            
        df = self.table.to_pandas()
        total_patches = 0
        
        stats = {
            'total_reports': len(df),
            'sections': {}
        }
        
        for section in self.SECTIONS:
            n_patches_col = f'{section}_n_patches'
            if n_patches_col in df.columns:
                section_total = df[n_patches_col].sum()
                non_empty = (df[n_patches_col] > 0).sum()
                total_patches += section_total
                
                stats['sections'][section] = {
                    'reports_with_patches': int(non_empty),
                    'total_patches': int(section_total)
                }
                
        stats['total_patches'] = int(total_patches)
        return stats

