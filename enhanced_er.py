"""
ENHANCED Entity Resolution Pipeline
Optimized for real-world multilingual data (Hindi, Kannada, Gujarati, Telugu, etc.)

Improvements over basic version:
- Empty address handling
- URL extraction from business names
- Better abbreviation handling
- Script-aware normalization
- Flag for empty/null addresses in features
"""

import pandas as pd
import numpy as np
import re
import unicodedata
from pathlib import Path
from typing import List, Tuple, Dict, Set
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# ENHANCED STRING NORMALIZER (with script/URL handling)
# ============================================================================

class EnhancedStringNormalizer:
    """Enhanced normalization for multilingual data"""
    
    # Extended legal suffixes
    LEGAL_SUFFIXES = {
        'pvt': 'pvt', 'private': 'pvt', 'ltd': 'ltd', 'limited': 'ltd',
        'inc': 'inc', 'incorporated': 'inc', 'corp': 'corp', 'corporation': 'corp',
        'llc': 'llc', 'llp': 'llp', 'co': 'co', 'company': 'co', 'pte': 'pte',
        'gmbh': 'gmbh', 'ag': 'ag', 'sa': 'sa',
        'sarl': 'sarl', 'sasu': 'sasu', 'eurl': 'eurl',  # French
    }
    
    # Extended abbreviations
    ADDR_ABBR = {
        'road': 'rd', 'rd': 'rd', 'street': 'st', 'st': 'st',
        'avenue': 'ave', 'ave': 'ave', 'boulevard': 'blvd', 'blvd': 'blvd',
        'building': 'bldg', 'bldg': 'bldg', 'floor': 'fl', 'fl': 'fl',
        'apartment': 'apt', 'apt': 'apt', 'suite': 'ste', 'ste': 'ste',
        'drive': 'dr', 'dr': 'dr', 'lane': 'ln', 'ln': 'ln',
        'court': 'ct', 'ct': 'ct', 'place': 'pl', 'pl': 'pl',
        'north': 'n', 'south': 's', 'east': 'e', 'west': 'w',
        'upper': 'u', 'lower': 'l', 'unit': 'u', 'no': 'no',
        'number': 'no', 'house': 'h', 'door': 'd',
    }
    
    # Hindi script transliteration (basic)
    HINDI_TRANSLITERATION = {
        'ा': 'a', 'ि': 'i', 'ी': 'ii', 'ु': 'u', 'ू': 'uu',
        'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', 'ँ': 'n',
        'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
        'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
        'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
        'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
        'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
        'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v',
        'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    }
    
    @staticmethod
    def is_empty_or_null(text: str) -> bool:
        """Check if text is empty or null string"""
        if pd.isna(text):
            return True
        text = str(text).strip()
        if text == '' or text.lower() in ['null', 'nan', 'none', '']:
            return True
        return False
    
    @staticmethod
    def clean_urls(text: str) -> str:
        """Remove URLs and web references from text"""
        if not text or pd.isna(text):
            return ''
        text = str(text)
        # Remove URLs
        text = re.sub(r'www\.\S+', '', text)
        text = re.sub(r'https?://\S+', '', text)
        # Remove pipes and domain separators
        text = re.sub(r'\|\s*', ' ', text)
        text = re.sub(r'\.com|\.in|\.org', '', text)
        return text.strip()
    
    @staticmethod
    def clean_special_prefixes(text: str) -> str:
        """Remove special prefixes and titles"""
        if not text or pd.isna(text):
            return ''
        text = str(text)
        # Remove leading dashes and special characters
        text = re.sub(r'^[\-\*\[\(]+', '', text)
        text = re.sub(r'[\)\]\*\|]+$', '', text)
        # Remove titles
        text = re.sub(r'\b(Dr\.|Mr\.|Mrs\.|Ms\.|Shri|Sri|Sri\.|Shri\.)\b', '', text, flags=re.IGNORECASE)
        return text.strip()
    
    @staticmethod
    def transliterate_hindi(text: str) -> str:
        """Simple Hindi-to-Latin transliteration"""
        if not text:
            return text
        for hindi, latin in EnhancedStringNormalizer.HINDI_TRANSLITERATION.items():
            text = text.replace(hindi, latin)
        return text
    
    @staticmethod
    def normalize(text: str, field_type: str = 'name') -> str:
        """Enhanced normalization for multilingual text"""
        
        # Check if empty/null
        if EnhancedStringNormalizer.is_empty_or_null(text):
            return ''
        
        text = str(text)
        
        # 1. Clean URLs and special prefixes
        text = EnhancedStringNormalizer.clean_urls(text)
        text = EnhancedStringNormalizer.clean_special_prefixes(text)
        
        # 2. Unicode normalization
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        
        # 3. Basic transliteration for Hindi (before lowercasing)
        text = EnhancedStringNormalizer.transliterate_hindi(text)
        
        # 4. Lowercase
        text = text.lower()
        
        # 5. Normalize whitespace
        text = ' '.join(text.split())
        
        # 6. Normalize punctuation and special chars
        text = text.replace('&', 'and')
        text = re.sub(r'[^\w\s]', '', text)
        
        # 7. Normalize legal suffixes
        for key, val in EnhancedStringNormalizer.LEGAL_SUFFIXES.items():
            text = re.sub(rf'\b{key}\b', val, text)
        
        # 8. Normalize address abbreviations (if address field)
        if field_type == 'address':
            for key, val in EnhancedStringNormalizer.ADDR_ABBR.items():
                text = re.sub(rf'\b{key}\b', val, text)
        
        # 9. Remove duplicate words (like "VIDYALAYA VIDYALAYA" → "vidyalaya")
        words = text.split()
        words = [w for i, w in enumerate(words) if i == 0 or w != words[i-1]]
        text = ' '.join(words)
        
        # 10. Final whitespace normalization
        text = ' '.join(text.split())
        
        return text


# ============================================================================
# ENHANCED FEATURE ENGINEER (with empty address flags)
# ============================================================================

class EnhancedFeatureEngineer:
    """Generate features with special handling for empty addresses"""
    
    @staticmethod
    def engineer_features(s1_record: dict, s2_record: dict) -> dict:
        """
        Engineer features with empty address detection
        """
        features = {}
        
        # Normalized strings
        n_name1 = EnhancedStringNormalizer.normalize(s1_record.get('business_name', ''), 'name')
        n_name2 = EnhancedStringNormalizer.normalize(s2_record.get('business_name', ''), 'name')
        n_addr1 = EnhancedStringNormalizer.normalize(s1_record.get('business_address', ''), 'address')
        n_addr2 = EnhancedStringNormalizer.normalize(s2_record.get('business_address', ''), 'address')
        
        # ---- NAME FEATURES (always computed) ----
        from entity_resolution import SimilarityMetrics  # Reuse from base module
        
        features['name_lev'] = SimilarityMetrics.normalized_levenshtein(n_name1, n_name2)
        features['name_jaccard'] = SimilarityMetrics.jaccard_similarity(n_name1, n_name2)
        features['name_jaro_winkler'] = SimilarityMetrics.jaro_winkler(n_name1, n_name2)
        features['name_char3gram'] = SimilarityMetrics.char_ngram_similarity(n_name1, n_name2, 3)
        features['name_char4gram'] = SimilarityMetrics.char_ngram_similarity(n_name1, n_name2, 4)
        
        # Length features
        name_len_ratio = len(n_name1) / (len(n_name2) + 1e-6)
        features['name_len_ratio'] = min(name_len_ratio, 1/name_len_ratio)
        
        # ---- ADDRESS FEATURES (with empty handling) ----
        has_addr1 = len(n_addr1) > 0
        has_addr2 = len(n_addr2) > 0
        
        features['addr_is_empty_s1'] = 0 if has_addr1 else 1
        features['addr_is_empty_s2'] = 0 if has_addr2 else 1
        features['both_addr_empty'] = 1 if (not has_addr1 and not has_addr2) else 0
        
        if has_addr1 and has_addr2:
            # Both have addresses → compute similarity
            features['addr_lev'] = SimilarityMetrics.normalized_levenshtein(n_addr1, n_addr2)
            features['addr_jaccard'] = SimilarityMetrics.jaccard_similarity(n_addr1, n_addr2)
            features['addr_char3gram'] = SimilarityMetrics.char_ngram_similarity(n_addr1, n_addr2, 3)
        else:
            # One or both empty → neutral (0.5) instead of 0.0
            features['addr_lev'] = 0.5
            features['addr_jaccard'] = 0.5
            features['addr_char3gram'] = 0.5
        
        # ---- COUNTRY MATCH ----
        features['country_match'] = 1.0 if s1_record.get('country') == s2_record.get('country') else 0.0
        
        # ---- COMBINED FEATURES ----
        # If either address empty, rely on name only
        if not has_addr1 or not has_addr2:
            features['name_addr_weighted'] = features['name_lev']
        else:
            features['name_addr_weighted'] = (features['name_lev'] * 0.6 + 
                                             features['addr_lev'] * 0.4)
        
        # ---- STRING LENGTH FEATURES ----
        features['name1_len'] = len(n_name1)
        features['name2_len'] = len(n_name2)
        features['addr1_len'] = len(n_addr1)
        features['addr2_len'] = len(n_addr2)
        
        # ---- TOKEN COUNT FEATURES ----
        features['name1_tokens'] = len(n_name1.split())
        features['name2_tokens'] = len(n_name2.split())
        features['addr1_tokens'] = len(n_addr1.split())
        features['addr2_tokens'] = len(n_addr2.split())
        
        # ---- PREFIX MATCH ----
        prefix_len = 3
        prefix1 = n_name1[:prefix_len] if len(n_name1) >= prefix_len else n_name1
        prefix2 = n_name2[:prefix_len] if len(n_name2) >= prefix_len else n_name2
        features['name_prefix_match'] = 1.0 if prefix1 == prefix2 else 0.0
        
        return features


# ============================================================================
# Enhanced Pipeline (drop-in replacement for EntityResolutionPipeline)
# ============================================================================

class EnhancedEntityResolutionPipeline:
    """Enhanced pipeline using improved normalization and features"""
    
    def __init__(self):
        # Import from base module to reuse
        from entity_resolution import Blocker as BaseBlocker
        self.BaseBlocker = BaseBlocker
        self.blocker = None
        self.model = None
        self.threshold = 0.5
        self.feature_names = None
    
    def load_data(self, data_dir: str = 'dataset/train'):
        """Load training data"""
        self.s1_train = pd.read_csv(f'{data_dir}/train_source1.tsv', sep='\t')
        self.s2_train = pd.read_csv(f'{data_dir}/train_source2.tsv', sep='\t')
        self.s3_train = pd.read_csv(f'{data_dir}/train_source3.tsv', sep='\t')
        self.ground_truth = pd.read_csv(f'{data_dir}/train_ground_truth.tsv', sep='\t')
        
        # Handle NaN in matched_entity_ids
        self.ground_truth['matched_entity_ids'] = self.ground_truth['matched_entity_ids'].fillna('')
        
        print(f"✓ Loaded {len(self.s1_train)} S1 records")
        print(f"✓ Loaded {len(self.s2_train)} S2 records")
        print(f"✓ Loaded {len(self.s3_train)} S3 records")
        print(f"✓ Loaded {len(self.ground_truth)} ground truth pairs")
        
        # Data quality check
        print(f"\n📊 Data Quality Check:")
        print(f"  S1 empty addresses: {self.s1_train['business_address'].isna().sum()}")
        print(f"  S2 empty addresses: {self.s2_train['business_address'].isna().sum()}")
        print(f"  S3 empty addresses: {self.s3_train['business_address'].isna().sum()}")
    
    def prepare_training_data(self):
        """Prepare training pairs using enhanced normalization"""
        # Use base blocker (strategies don't need to change, just output quality improves)
        self.blocker = self.BaseBlocker(self.s2_train, self.s3_train)
        
        # Update normalized names in blocker to use enhanced normalization
        self.s2_train['norm_name'] = self.s2_train['business_name'].apply(
            lambda x: EnhancedStringNormalizer.normalize(x, 'name')
        )
        self.s3_train['norm_name'] = self.s3_train['business_name'].apply(
            lambda x: EnhancedStringNormalizer.normalize(x, 'name')
        )
        
        # ⭐ CRITICAL FIX: Rebuild blocker indexes with the NEW normalized names
        # Without this, the blocker has stale/empty indexes and returns 0 candidates
        self.blocker._build_indexes()
        
        # Combine S2 and S3
        self.s2_train['source'] = 2
        self.s3_train['source'] = 3
        self.combined = pd.concat([self.s2_train, self.s3_train], ignore_index=True)
        self.entity_to_idx = {row['entity_id']: i for i, row in self.combined.iterrows()}
        
        # Generate training pairs
        X = []
        y = []
        
        for idx, s1_row in self.s1_train.iterrows():
            s1_id = s1_row['entity_id']
            
            # Get ground truth matches
            gt_row = self.ground_truth[self.ground_truth['source1_entity_id'] == s1_id]
            
            if len(gt_row) == 0:
                matched_ids = set()
            else:
                matched_str = gt_row.iloc[0]['matched_entity_ids']
                matched_ids = set(str(matched_str).split(',')) if matched_str and matched_str != '' else set()
            
            # Generate candidates
            candidates = self.blocker.generate_candidates(dict(s1_row))
            
            # Create pairs with enhanced features
            for cand_idx in candidates:
                cand_row = self.combined.iloc[cand_idx]
                cand_id = cand_row['entity_id']
                
                features = EnhancedFeatureEngineer.engineer_features(dict(s1_row), dict(cand_row))
                X.append(features)
                y.append(1 if cand_id in matched_ids else 0)
            
            if idx % 100 == 0:
                print(f"  Processed {idx}/{len(self.s1_train)} S1 records...")
        
        self.X_train = pd.DataFrame(X)
        self.y_train = np.array(y)
        self.feature_names = self.X_train.columns.tolist()
        
        print(f"\n✓ Generated {len(self.X_train)} training pairs")
        print(f"  Positive: {self.y_train.sum()}, Negative: {(1-self.y_train).sum()}")
        print(f"  Positive rate: {self.y_train.mean():.2%}")
    
    def train_model(self):
        """Train LightGBM model"""
        try:
            import lightgbm as lgb
        except ImportError:
            print("⚠ LightGBM not installed. Install with: pip install lightgbm")
            return
        
        pos_count = self.y_train.sum()
        neg_count = len(self.y_train) - pos_count
        scale_pos_weight = neg_count / max(pos_count, 1)
        
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            num_leaves=31,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1,
        )
        
        self.model.fit(self.X_train, self.y_train)
        print(f"✓ Model trained with {len(self.feature_names)} features")
        
        # Show feature importance
        importances = self.model.feature_importances_
        top_features = sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n  Top 5 features:")
        for fname, fimportance in top_features:
            print(f"    {fname}: {fimportance:.3f}")
    
    def predict_on_test(self, test_dir: str = 'dataset/test'):
        """Generate predictions on test set"""
        s1_test = pd.read_csv(f'{test_dir}/test_source1.tsv', sep='\t')
        s2_test = pd.read_csv(f'{test_dir}/test_source2.tsv', sep='\t')
        s3_test = pd.read_csv(f'{test_dir}/test_source3.tsv', sep='\t')
        
        # Combine S2 and S3
        s2_test['source'] = 2
        s3_test['source'] = 3
        combined_test = pd.concat([s2_test, s3_test], ignore_index=True)
        
        # Create blocker for test data
        blocker_test = self.BaseBlocker(s2_test, s3_test)
        # Update normalized names
        s2_test['norm_name'] = s2_test['business_name'].apply(
            lambda x: EnhancedStringNormalizer.normalize(x, 'name')
        )
        s3_test['norm_name'] = s3_test['business_name'].apply(
            lambda x: EnhancedStringNormalizer.normalize(x, 'name')
        )
        blocker_test.source2 = s2_test
        blocker_test.source3 = s3_test
        blocker_test._build_indexes()
        
        matching_results = []
        candidate_pairs = []
        
        for idx, s1_row in s1_test.iterrows():
            s1_id = s1_row['entity_id']
            
            # Generate candidates
            candidates_idx = blocker_test.generate_candidates(dict(s1_row))
            
            if not candidates_idx:
                matching_results.append({'source1_entity_id': s1_id, 'matched_entity_ids': ''})
                candidate_pairs.append({'source1_entity_id': s1_id, 'candidate_entity_ids': ''})
                continue
            
            # Generate features and predict
            X_cand = []
            cand_ids = []
            
            for cand_idx in candidates_idx:
                cand_row = combined_test.iloc[cand_idx]
                features = EnhancedFeatureEngineer.engineer_features(dict(s1_row), dict(cand_row))
                
                X_cand.append([features.get(fn, 0) for fn in self.feature_names])
                cand_ids.append(cand_row['entity_id'])
            
            X_cand = np.array(X_cand)
            
            # Get predictions
            if self.model is not None:
                probs = self.model.predict_proba(X_cand)[:, 1]
            else:
                # Fallback: use simple name+address similarity
                probs = []
                for features in X_cand:
                    score = features[0]  # name_lev
                    probs.append(score)
                probs = np.array(probs)
            
            # Get matches above threshold
            matches = [cand_ids[i] for i in range(len(cand_ids)) if probs[i] >= self.threshold]
            
            # Store results
            matched_str = ','.join(matches) if matches else ''
            matching_results.append({'source1_entity_id': s1_id, 'matched_entity_ids': matched_str})
            
            candidate_str = ','.join(cand_ids) if cand_ids else ''
            candidate_pairs.append({'source1_entity_id': s1_id, 'candidate_entity_ids': candidate_str})
            
            if idx % 1000 == 0:
                print(f"  Predicted {idx}/{len(s1_test)} test records...")
        
        self.matching_results_df = pd.DataFrame(matching_results)
        self.candidate_pairs_df = pd.DataFrame(candidate_pairs)
        
        print(f"✓ Generated predictions for {len(s1_test)} test records")
    
    def save_results(self, output_dir: str = 'output'):
        """Save results to TSV files"""
        Path(output_dir).mkdir(exist_ok=True)
        
        self.matching_results_df.to_csv(
            f'{output_dir}/matching_results.tsv',
            sep='\t',
            index=False,
            quoting=3
        )
        
        self.candidate_pairs_df.to_csv(
            f'{output_dir}/candidate_pairs.tsv',
            sep='\t',
            index=False,
            quoting=3
        )
        
        print(f"✓ Results saved to {output_dir}/")
    
    def run_enhanced_pipeline(self, train_dir='dataset/train', test_dir='dataset/test', output_dir='output'):
        """Run enhanced pipeline"""
        print("\n" + "="*70)
        print("ENHANCED ENTITY RESOLUTION PIPELINE")
        print("="*70)
        
        print("\n[1/5] Loading training data...")
        self.load_data(train_dir)
        
        print("\n[2/5] Preparing training data...")
        self.prepare_training_data()
        
        print("\n[3/5] Training model...")
        self.train_model()
        
        print("\n[4/5] Predicting on test set...")
        self.predict_on_test(test_dir)
        
        print("\n[5/5] Saving results...")
        self.save_results(output_dir)
        
        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETE")
        print("="*70)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    pipeline = EnhancedEntityResolutionPipeline()
    pipeline.run_enhanced_pipeline()