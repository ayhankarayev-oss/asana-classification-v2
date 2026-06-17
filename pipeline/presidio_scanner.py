"""
Presidio-based PII detection using Microsoft Presidio ML models.
Detects person names, organizations, and other PII in freeform text.
"""
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from typing import Optional


class PresidioScanner:
    """
    ML-based PII detection using Microsoft Presidio.
    Detects PERSON names, ORGANIZATION names, and other PII that regex patterns miss.
    """
    
    def __init__(self, confidence_threshold: float = 0.85):
        """
        Initialize Presidio analyzer and anonymizer.
        
        Args:
            confidence_threshold: Minimum confidence score (0.0-1.0) to accept detections.
                                 Higher = fewer false positives. Default 0.85.
        """
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.confidence_threshold = confidence_threshold
        
        # Entity types to detect
        self.entity_types = [
            "PERSON",           # Person names (Mike Albanese, Carolina Letsos)
            "EMAIL_ADDRESS",    # Backup for emails
            "PHONE_NUMBER",     # Backup for phones
            "ORGANIZATION",     # Company names (Mosaic Advisors, etc.)
            "US_SSN",          # SSN backup
            "CREDIT_CARD",     # Credit card backup
        ]
    
    def scan_and_mask(self, text: str, mask_location: bool = False) -> tuple[str, list]:
        """
        Scan text for PII using ML models and return masked text + detection log.
        
        Args:
            text: Input text to scan
            mask_location: If True, also mask LOCATION entities (Houston, TX, etc.)
                          Default False - locations may be useful operational context
        
        Returns:
            (masked_text, detections_list)
            
        Example:
            scanner = PresidioScanner()
            text = "Mike Albanese reported an issue with Mosaic Advisors."
            masked, log = scanner.scan_and_mask(text)
            # masked: "[PERSON] reported an issue with [ORG]."
            # log: [{"entity_type": "PERSON", "text": "Mike Albanese", ...}, ...]
        """
        if not text or not text.strip():
            return text, []
        
        # Build entity list
        entities_to_detect = self.entity_types.copy()
        if mask_location:
            entities_to_detect.append("LOCATION")
        
        # Analyze text for PII
        try:
            results = self.analyzer.analyze(
                text=text,
                entities=entities_to_detect,
                language="en"
            )
        except Exception as e:
            # If Presidio fails, return original text (don't break pipeline)
            print(f"Presidio analysis failed: {e}")
            return text, []
        
        # Filter by confidence threshold
        results = [r for r in results if r.score >= self.confidence_threshold]
        
        # Build detection log
        detections = []
        for result in results:
            detections.append({
                "entity_type": result.entity_type,
                "start": result.start,
                "end": result.end,
                "score": result.score,
                "text": text[result.start:result.end]
            })
        
        # Anonymize detected entities
        if results:
            try:
                from presidio_anonymizer.entities import OperatorConfig
                
                anonymized = self.anonymizer.anonymize(
                    text=text,
                    analyzer_results=results,
                    operators={
                        "PERSON": OperatorConfig("replace", {"new_value": "[PERSON]"}),
                        "ORGANIZATION": OperatorConfig("replace", {"new_value": "[ORG]"}),
                        "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
                        "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE]"}),
                        "LOCATION": OperatorConfig("replace", {"new_value": "[LOCATION]"}),
                        "US_SSN": OperatorConfig("replace", {"new_value": "[ID]"}),
                        "CREDIT_CARD": OperatorConfig("replace", {"new_value": "[CREDIT_CARD]"}),
                        "DEFAULT": OperatorConfig("replace", {"new_value": "[PII]"}),
                    }
                )
                return anonymized.text, detections
            except Exception as e:
                print(f"Presidio anonymization failed: {e}")
                return text, detections
        
        return text, detections


# Convenience function for one-off scans
def scan_text(text: str, confidence: float = 0.85, mask_location: bool = False) -> tuple[str, list]:
    """
    Convenience function to scan a single text without instantiating scanner.
    
    Args:
        text: Text to scan
        confidence: Confidence threshold (0.0-1.0)
        mask_location: Whether to mask location entities
    
    Returns:
        (masked_text, detections)
    """
    scanner = PresidioScanner(confidence_threshold=confidence)
    return scanner.scan_and_mask(text, mask_location=mask_location)
