import os
import sys
import csv
import json
import logging
import argparse
from enum import Enum
from collections import defaultdict
from json import JSONDecodeError
from typing import Optional, Any, Dict, Tuple, List
from datetime import datetime
from openai import OpenAI, OpenAIError
from functools import lru_cache

# Environment configurations

CURRENT_DATE = "2025-07-30"
RESUBMISSION_WINDOW = 7
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level = LOG_LEVEL,
    format = "%(asctime)s - %(levelname)s - %(message)s",
    handlers = [logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

class ClaimStatus(Enum):
    APPROVED = "approved"
    DENIED = "denied"

class DenialCategory(Enum):
    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"
    AMBIGUOUS = "ambiguous"

class FileType(Enum):
    CSV = "csv"
    JSON = "json"

class ClaimsProcessor:
    "Processes insurance claims from multiple EMR Sources"
    def __init__(self):
        self.metrics = {
            "total_claims" : 0,
            "resubmission_candidates" : 0,
            "source_counts" : defaultdict(int),
            "exclusion_reasons" : defaultdict(int),
        }

        self.hardcoded_recommendations = {
                "missing modifier": "Add appropriate modifier and resubmit",
                "incorrect npi": "Verify provider NPI and resubmit",
                "prior auth required": "Obtain prior authorization and resubmit",
                "incorrect procedure": "Review procedure coding guidelines",
                "form incomplete": "Complete all required form fields",
                "not billable": "Verify billing eligibility requirements",
            }

        self.rejected_claims = []

    def normalize_field_names(self, row: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Standardize field names across sources"""
        field_map = {
            "alpha": {
                "claim_id": "claim_id",
                "patient_id": "patient_id",
                "procedure_code": "procedure_code",
                "denial_reason": "denial_reason",
                "status": "status",
                "submitted_at": "submitted_at",
            },
            "beta": {
                "claim_id": "id",
                "patient_id": "member",
                "procedure_code": "code",
                "denial_reason": "error_msg",
                "status": "status",
                "submitted_at": "date",
            }
        }
        return {target: row.get(source_field) 
                for target, source_field in field_map[source].items()}

    def clean_value(self, value: Any) -> Optional[str]:
        """Handle nulls and formatting consistently"""
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return None
        if isinstance(value, str):
            return value.strip().lower()
        return str(value)

    def parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Robust date parsing with multiple format support"""
        if not date_str:
            return None
            
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%m/%d/%Y"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
        logger.warning(f"Unparseable date format: {date_str}")
        return None

    def normalize_row(self, raw_row: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Transform raw data into standardized schema"""
        row = self.normalize_field_names(raw_row, source)
        normalized = {}
        for key, value in row.items():
            if key in ("claim_id", "patient_id"):
                # Preserve original capitalization/formatting for IDs
                if value is None or (isinstance(value, str) and value.strip() == ""):
                    normalized[key] = None
                elif isinstance(value, str):
                    normalized[key] = value.strip()
                else:
                    normalized[key] = str(value)
            else:
                normalized[key] = self.clean_value(value)
        
        # Special handling for dates
        if submitted_at := normalized.get("submitted_at"):
            normalized["submitted_at"] = self.parse_date(submitted_at)
        
        # Add source system identifier
        normalized["source_system"] = source
        
        return normalized

    def validate_claim(self, claim: Dict[str, Any]) -> bool:
        """Check required fields exist"""
        required_fields = ["claim_id", "status", "submitted_at"]
        return all(claim.get(field) is not None for field in required_fields)

    def classify_denial_reason(self, reason: Optional[str]) -> Tuple[DenialCategory, str]:
        """Categorize denial reason with hardcoded rules"""
        if not reason:
            return DenialCategory.AMBIGUOUS, "No reason provided"
        
        reason_lower = reason.lower()
        
        # Known retryable reasons
        RETRYABLE = {
            "missing modifier", 
            "incorrect npi", 
            "prior auth required"
        }
        if reason_lower in RETRYABLE:
            return DenialCategory.RETRYABLE, reason
        
        # Known non-retryable reasons
        NON_RETRYABLE = {
            "authorization expired", 
            "incorrect provider type"
        }
        if reason_lower in NON_RETRYABLE:
            return DenialCategory.NON_RETRYABLE, reason
        
        return DenialCategory.AMBIGUOUS, reason

    @lru_cache(maxsize=100)
    def generate_recommendation(self, reason: Optional[str]) -> str:
        """Generate resubmission recommendation with caching"""
        if not reason:
            return "Review claim documentation for errors"
        
        # Check hardcoded recommendations first
        reason_lower = reason.lower()
        for pattern, recommendation in self.hardcoded_recommendations.items():
            if pattern in reason_lower:
                return recommendation
        
        # Fallback to LLM if available
        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.responses.create(
                model = LLM_MODEL,
                input = f"""
                {{
                    "claim_id": "A124",
                    "resubmission_reason": "Incorrect NPI",
                    "source_system": "alpha",
                    "recommended_changes": "Review NPI number and resubmit"
                }}

                Based on the example above return a recommended_changes for 
                the denial reason below:
                {denial_reason}

                Return the response strictly as JSON:
                {{"recommended_changes": "<your recommended change>"}}
                """
            )
        
            return json.loads(response.output_text)['recommended_changes']
        except ImportError:
            logger.debug("OpenAI library not available")
        except Exception as e:
            logger.error(f"LLM recommendation failed: {str(e)}")
        
        # Final fallback
        return f"Investigate reason: {reason}"

    def is_eligible_for_resubmission(self, claim: Dict[str, Any]) -> bool:
        """Determine if claim meets all resubmission criteria according to business rules"""
        rejection_reason = None
        
        # 1. Status must be denied
        if claim["status"] != ClaimStatus.DENIED.value:
            rejection_reason = "Claim status is approved"
            self._track_rejection(claim, rejection_reason, "status_approved")
            return False
        
        # 2. Patient ID must exist
        if not claim.get("patient_id"):
            rejection_reason = "Missing patient_id"
            self._track_rejection(claim, rejection_reason, "patient_id_null")
            return False
        
        # 3. Must be submitted more than 7 days ago
        if not claim["submitted_at"] or (self.parse_date(CURRENT_DATE) - claim["submitted_at"]).days <= RESUBMISSION_WINDOW:
            rejection_reason = f"Submitted within last {RESUBMISSION_WINDOW} days"
            self._track_rejection(claim, rejection_reason, "submitted_recently")
            return False
        
        # 4. Denial reason handling
        denial_reason = claim.get("denial_reason")
        normalized_reason = denial_reason.lower() if denial_reason else None
        
        # Hardcoded reason sets
        HARDCODED_RETRYABLE = {
            "missing modifier",
            "incorrect npi",
            "prior auth required"
        }
        
        HARDCODED_NON_RETRYABLE = {
            "authorization expired",
            "incorrect provider type"
        }
        
        # Check against known lists
        if normalized_reason in HARDCODED_NON_RETRYABLE:
            rejection_reason = f"Non-retryable reason: {denial_reason}"
            self._track_rejection(claim, rejection_reason, "non_retryable_reason")
            return False
        
        if normalized_reason in HARDCODED_RETRYABLE:
            return True
        
        # Handle ambiguous cases
        retry_ambiguous = self.should_retry_ambiguous_reason(normalized_reason)
        if retry_ambiguous is True:
            return True
        elif retry_ambiguous is False:
            rejection_reason = f"Ambiguous reason not approved for resubmission: {denial_reason}"
            self._track_rejection(claim, rejection_reason, "ambiguous_reason")
            return False
        else:
            rejection_reason = f"Unclassifiable reason: {denial_reason}"
            self._track_rejection(claim, rejection_reason, "unclassifiable_reason")
            return False

    def _track_rejection(self, claim: Dict[str, Any], reason: str, metric_key: str) -> None:
        """Track rejected claims and update metrics"""
        self.metrics["exclusion_reasons"][metric_key] += 1
        self.metrics["total_rejected"] = self.metrics.get("total_rejected", 0) + 1
        
        rejected_claim = {
            "claim_id": claim.get("claim_id").upper() if claim['claim_id'] is not None else None,
            "patient_id": claim.get("patient_id").upper() if claim.get("patient_id") is not None else None,
            "procedure_code": claim.get("procedure_code"),
            "denial_reason": claim.get("denial_reason"),
            "submitted_at": claim["submitted_at"].isoformat() if claim.get("submitted_at") else None,
            "source_system": claim.get("source_system"),
            "rejection_reason": reason,
            "rejection_category": metric_key
        }
        
        self.rejected_claims.append(rejected_claim)

    def should_retry_ambiguous_reason(self, reason: Optional[str]) -> bool:
        """Determine if ambiguous reason should be retried using heuristic/LLM"""
        # Hardcoded heuristic for ambiguous reasons
        AMBIGUOUS_RETRY_HEURISTIC = {
            "incorrect procedure": True,  # Retry
            "form incomplete": True,      # Retry
            "not billable": False,        # Don't retry
            None: False                   # Don't retry
        }
        
        # First try hardcoded mapping
        if reason in AMBIGUOUS_RETRY_HEURISTIC:
            return AMBIGUOUS_RETRY_HEURISTIC[reason]
        
        # Fallback to LLM 
        # We will use a mock LLM implementation 
        try:
            return self.llm_classify_reason(reason)
        except Exception as e:
            logger.warning(f"LLM classification failed: {str(e)}")
            return False  # Default to not retry if classification fails

    def llm_classify_reason(self, reason: Optional[str]) -> bool:
        """Mock LLM classifier for ambiguous reasons"""
        # In a real implementation, this would call an actual LLM API
        # This is a simplified mock that returns True for reasons containing "incomplete" or "form"
        if not reason:
            return False
        
        keywords = ["incomplete", "form"]
        return any(keyword in reason.lower() for keyword in keywords)

    def process_row(self, row: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
        """Process a single claim record"""
        self.metrics["total_claims"] += 1
        self.metrics["source_counts"][source] += 1
        
        try:
            normalized = self.normalize_row(row, source)
            
            if not self.validate_claim(normalized):
                self.metrics["exclusion_reasons"]["invalid_data"] += 1
                return None
                
            if not self.is_eligible_for_resubmission(normalized):
                return None
                
            self.metrics["resubmission_candidates"] += 1
            
            return {
                "claim_id": normalized["claim_id"],
                "resubmission_reason": normalized["denial_reason"],
                "source_system": source,
                "recommended_changes": self.generate_recommendation(
                    normalized["denial_reason"]
                )
            }
        except Exception as e:
            print(row)
            logger.error(f"Error processing claim: {str(e)}")
            self.metrics["exclusion_reasons"]["processing_error"] += 1
            return None

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process all claims in a single file"""
        eligible_claims = []
        
        try:
            if file_path.endswith(FileType.CSV.value):
                with open(file_path, "r", encoding="utf-8") as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        if processed := self.process_row(row, "alpha"):
                            eligible_claims.append(processed)
            
            elif file_path.endswith(FileType.JSON.value):
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    if not isinstance(data, list):
                        data = [data]
                    for row in data:
                        if processed := self.process_row(row, "beta"):
                            eligible_claims.append(processed)
            else:
                logger.error(f"Unsupported file type: {file_path}")
                
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {str(e)}")
            raise
        
        return eligible_claims

    def process_directory(self, directory: str) -> List[Dict[str, Any]]:
        """Process all EMR files in a directory"""
        all_eligible = []
        
        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            return all_eligible
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                all_eligible.extend(self.process_file(file_path))
        
        return all_eligible


    def generate_report(self) -> Dict[str, Any]:
        """Generate processing metrics report"""
        return {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "current_date": self.parse_date(CURRENT_DATE).isoformat(),
                "resubmission_window_days": RESUBMISSION_WINDOW,
                "llm_model": LLM_MODEL
            },
            "metrics": self.metrics
        }

def main():
    """Command-line interface for claim processing"""
    parser = argparse.ArgumentParser(description="Claim Resubmission Pipeline")
    parser.add_argument("directory", help="Directory containing EMR files")
    parser.add_argument("-o", "--output", default="resubmission_candidates.json",
                        help="Output file path")
    parser.add_argument("-r", "--report", default="processing_report.json",
                        help="Metrics report file path")
    parser.add_argument("-e", "--errors", default="rejected_claims.log",
                        help="Error log file path")
    parser.add_argument("-re", "--rejected", default="rejected_claims.json",
                        help="Rejected claims file path")
    args = parser.parse_args()

    # Setup error logging to file
    file_handler = logging.FileHandler(args.errors)
    file_handler.setLevel(logging.ERROR)
    logger.addHandler(file_handler)

    processor = ClaimsProcessor()
    eligible_claims = processor.process_directory(args.directory)
    rejected_claims = processor.rejected_claims
    
    # Save eligible claims
    with open(args.output, "w") as file:
        json.dump(eligible_claims, file, indent=4)

    # Save rejected claims
    with open(args.rejected, 'w') as file:
        json.dump(rejected_claims, file, indent=4)

    # Save metrics report
    with open(args.report, "w") as file:
        json.dump(processor.generate_report(), file, indent=4)
    
    # Print summary
    logger.info(f"Processed {processor.metrics['total_claims']} claims")
    logger.info(f"Found {len(eligible_claims)} resubmission candidates")
    logger.info(f"Found {len(rejected_claims)} rejected candidates")
    logger.info(f"Results on resubmission candidates saved to {args.output}")
    logger.info(f"Results on rejected candidates saved to {args.rejected}")
    logger.info(f"Error logs saved to {args.errors}")
    logger.info(f"Processing reports and metrics saved to {args.report}")

if __name__ == "__main__":
    main()




