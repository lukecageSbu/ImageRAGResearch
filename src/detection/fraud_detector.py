import os
import json
import base64
from typing import Dict, List, Tuple

from configs import FraudDetectionConfig

class InvestigationEngine:
    def __init__(self, config: FraudDetectionConfig = None):
        self.config = config or FraudDetectionConfig()
        self.bedrock = self.config.get_bedrock_client()
        self.model_id = self.config.INVESTIGATOR_MODEL

    def detect_fraud(self, 
                     query_image_path: str, 
                     similar_case_image_paths: List[tuple]) -> Tuple[str, float, str]:
        
        section_cases = {}
        for img_path, similarity, section in similar_case_image_paths:
            if section not in section_cases:
                section_cases[section] = []
            section_cases[section].append((img_path, similarity))
            
        available_sections = list(section_cases.keys())
        
        if len(available_sections) < 3:
            return (
                "FAIL",
                1.0,
                f"Insufficient section coverage: only {len(available_sections)} of 5 sections available."
            )
            
        section_results = {}
        failed_sections = []
        
        for section in ['header', 'items', 'payment', 'metadata', 'footer']:
            if section not in section_cases:
                continue
                
            cases = sorted(section_cases[section], key=lambda x: x[1], reverse=True)
            
            prompt = self._build_fraud_detection_prompt(query_image_path, cases)
            response = self._call_llm(prompt)
            status, confidence, reasoning = self._parse_response(response)
            
            section_results[section] = {
                'status': status,
                'confidence': confidence,
                'reasoning': reasoning
            }
            
            if status == "FAIL":
                failed_sections.append(section)
                
        if failed_sections:
            overall_status = "FAIL"
            overall_confidence = max(r['confidence'] for r in section_results.values() if r['status'] == 'FAIL')
            # Include detailed reasoning from each failed section
            section_details = []
            for section in failed_sections:
                section_reasoning = section_results[section]['reasoning']
                section_details.append(f"[{section.upper()}] {section_reasoning}")
            
            overall_reasoning = f"Fraud detected in {len(failed_sections)} section(s): {', '.join(failed_sections)}. " + " ".join(section_details)
        else:
            overall_status = "PASS"
            confidences = [r['confidence'] for r in section_results.values() if r['status'] == 'PASS']
            overall_confidence = min(confidences) if confidences else 0.5
            overall_reasoning = f"All {len(section_results)} sections passed fraud checks"
            
        return overall_status, overall_confidence, overall_reasoning

    def _build_fraud_detection_prompt(self, query_image_path: str, similar_case_image_paths: List[tuple]) -> List[Dict]:
        content = []
        
        self._validate_image(query_image_path)
        
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": self._encode_image(query_image_path)
            }
        })
        
        content.append({
            "type": "text",
            "text": "\nREPORT UNDER INVESTIGATION\n"
        })
        
        valid_case_count = 0
        for i, (image_path, similarity) in enumerate(similar_case_image_paths, 1):
            if self._is_valid_image(image_path):
                valid_case_count += 1
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": self._encode_image(image_path)
                    }
                })
                
                content.append({
                    "type": "text",
                    "text": f"\nSIMILAR CASE {valid_case_count} (Similarity: {similarity:.2f})\n"
                })

        content.append({
            "type": "text",
            "text": self._get_fraud_detection_instructions(valid_case_count)
        })
        
        return content

    def _get_fraud_detection_instructions(self, valid_case_count: int) -> str:
        return f"""You are an expert forensic document examiner specializing in receipt and invoice fraud.

YOUR TASK: Verify if the "REPORT UNDER INVESTIGATION" is a legitimate receipt or a forgery/fabrication.
You have been shown {valid_case_count} similar authentic receipts for comparison.

------------------------------------------
RECEIPT FRAUD DETECTION ANALYSIS
------------------------------------------

# ANALYSIS FRAMEWORK

## 1. DIGITAL ALTERATION & TEMPLATE ABUSE
**Detection Criteria:**
- **Font Mismatches:** Multiple fonts in the same section (e.g., items vs price), or font size/weight inconsistencies that suggest digital insertion.
- **Misalignment:** Text that floats or doesn't align with the column grid (especially in the 'Items' or 'Prices' columns).
- **Artifacts:** Pixelation around specific numbers (Total, Date) while other text is sharp.
- **Template Artifacts:** Visible bounding boxes, placeholder text (e.g., "Company Name Here"), or generic logos.

## 2. ARITHMETIC & LOGIC VALIDATION
**Detection Criteria:**
- **Math Check:** Do the individual item prices sum up to the Subtotal?
- **Tax Check:** Is the tax (GST/SST/VAT) calculated correctly based on the Subtotal? (e.g., 6% or 10%).
- **Rounding:** Is the rounding adjustment logically correct?
- **Consistency:** Does "Cash" tendered minus "Change" equal the "Grand Total"?

## 3. MERCHANT IDENTITY & LAYOUT CONSISTENCY
**Detection Criteria:**
- **Logo/Header:** Compare with the provided similar cases. Is the logo distorted, wrong aspect ratio, or lower quality?
- **Address/Details:** Are there typos in the merchant's address or phone number?
- **Layout:** Does the receipt layout (columns, separators, font style) match the similar cases from the same merchant type?

## 4. CONTENT PLAUSIBILITY
**Detection Criteria:**
- **Time/Date:** Is the time logical? (e.g., shopping mall receipt at 3:00 AM).
- **Items:** Are the items consistent with the merchant type? (e.g., "Fuel" appearing on a bookstore receipt).

------------------------------------------
ANALYSIS OUTPUT REQUIREMENTS
------------------------------------------
Provide a comprehensive fraud detection report.
If you detect ANY indicators from the framework above, mark the status as **FAIL**.

REQUIRED OUTPUT FORMAT (PLAIN TEXT):
Status: [PASS | FAIL]
Confidence: [0.0-1.0] (High confidence = clear evidence of fraud or clear evidence of authenticity)
Reasoning: [Detailed explanation of findings]
Evidence: [Specific examples: "Total 88.90 does not match sum of items 50.00 + 20.00", "Font for date is Arial while rest is Dot Matrix", etc.]
"""

    def _validate_image(self, image_path: str):
        from PIL import Image
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        file_size = os.path.getsize(image_path)
        if file_size > 5_242_880:
            raise ValueError(f"File size {file_size} bytes exceeds 5 MB limit")
            
        with Image.open(image_path) as img:
            width, height = img.size
            if width > 8000 or height > 8000:
                raise ValueError(f"Dimensions {width}x{height} exceed 8000 pixel limit")

    def _is_valid_image(self, image_path: str) -> bool:
        try:
            self._validate_image(image_path)
            return True
        except (FileNotFoundError, ValueError):
            return False

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _call_llm(self, content: List[Dict]) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.1
        }
        
        response = self.bedrock.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body)
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']

    def _parse_response(self, response: str) -> Tuple[str, float, str]:
        lines = response.strip().split('\n')
        
        status = "UNKNOWN"
        confidence = 0.0
        reasoning = ""
        evidence = ""
        
        i = 0
        while i < len(lines):
            line = lines[i].strip().replace('**', '').replace('##', '').replace('#', '').strip()
            
            if "status:" in line.lower():
                status_text = line.split(':', 1)[1].strip().upper()
                if "PASS" in status_text:
                    status = "PASS"
                elif "FAIL" in status_text:
                    status = "FAIL"
                    
            elif "confidence:" in line.lower():
                try:
                    conf_text = line.split(':', 1)[1].strip().split()[0]
                    confidence = float(conf_text)
                except (ValueError, IndexError):
                    confidence = 0.5
                    
            elif "reasoning:" in line.lower():
                # Capture reasoning - may be multi-line until Evidence: or end
                reasoning_parts = [line.split(':', 1)[1].strip()]
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if next_line.lower().startswith('evidence:') or next_line.lower().startswith('status:'):
                        i -= 1 # Back up so outer loop processes this line
                        break
                    if next_line:
                        reasoning_parts.append(next_line)
                    i += 1
                reasoning = " ".join(reasoning_parts)
                
            elif "evidence:" in line.lower():
                # Capture evidence - may be multi-line until end or next field
                evidence_parts = [line.split(':', 1)[1].strip()]
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if next_line.lower().startswith('status:') or next_line.lower().startswith('reasoning:'):
                        i -= 1
                        break
                    if next_line:
                        evidence_parts.append(next_line)
                    i += 1
                evidence = " ".join(evidence_parts)
                
            i += 1
            
        # Combine reasoning and evidence
        full_reasoning = reasoning
        if evidence and evidence.lower() != "no fraud indicators detected":
            full_reasoning = f"{reasoning} Evidence: {evidence}"
            
        if status == "UNKNOWN":
            full_reasoning = f"UNKNOWN - LLM response malformed. Raw: {response[:200]}"
            
        return status, confidence, full_reasoning

