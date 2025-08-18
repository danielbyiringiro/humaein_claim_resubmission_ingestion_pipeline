from prefect import flow, task, get_run_logger
from main import ClaimsProcessor
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import json
import os

@task(
    retries=3,
    retry_delay_seconds=10,
    log_prints=True,
    timeout_seconds=300
)
def extract_claims(file_path: str) -> List[Dict]:
    """Extract and process claims from a file"""
    logger = get_run_logger()
    try:
        logger.info(f"Processing file: {file_path}")
        processor = ClaimsProcessor()
        claims = processor.process_file(file_path)
        logger.info(f"Successfully processed {len(claims)} raw claims")
        return claims
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {str(e)}")
        raise

@task(
    cache_key_fn=lambda *_args, **_kwargs: str(datetime.today().date()),  # Cache per day
    cache_expiration=timedelta(days=1)
)
def filter_eligible(claims: List[Dict]) -> List[Dict]:
    """Filter eligible claims for resubmission"""
    logger = get_run_logger()
    eligible = [claim for claim in claims if claim is not None]
    logger.info(f"Found {len(eligible)} eligible claims")
    return eligible

@task
def load_results(claims: List[Dict]):
    """Save results with proper error handling"""
    logger = get_run_logger()
    try:
        # Create output directory if it doesn't exist
        os.makedirs("output", exist_ok=True)
        
        output_path = f"output/resubmission_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Save with atomic write to prevent corruption
        temp_path = f"{output_path}.tmp"
        with open(temp_path, "w") as f:
            json.dump(claims, f, indent=4)
        os.rename(temp_path, output_path)
        
        logger.info(f"Saved {len(claims)} claims to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        raise

@flow(
    name="Claims Resubmission Pipeline",
    description="Processes EMR claims and identifies resubmission candidates",
    version="1.0.0",
    timeout_seconds=3600
)
def claims_resubmission_flow(file_path: str = "emr_files/emr_alpha.csv"):
    """Main workflow for claim resubmission processing"""
    logger = get_run_logger()
    try:
        logger.info("Starting claims resubmission pipeline")
        
        # Process claims
        raw_claims = extract_claims(file_path)
        eligible_claims = filter_eligible(raw_claims)
        result_path = load_results(eligible_claims)
        
        logger.info(f"Pipeline completed successfully. Results at {result_path}")
        return result_path
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    # For testing
    claims_resubmission_flow()
    