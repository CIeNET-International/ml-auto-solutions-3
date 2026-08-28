import re
from datetime import datetime, timezone
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from dags.orbax.util.validation_util import list_log_entries
from google.cloud import logging as logging_api

@task
def validate_semantic_inference_output(
    project_id: str,
    location: str,
    cluster_name: str,
    start_time: str,  # Passed as UTC ISO string from airflow context (e.g. "{{ ts }}")
) -> None:
    """
    Validates the inference output from Cloud Logging using regex patterns.
    Throws AirflowFailException on gibberish or invalid text.
    """
    start_dt = datetime.fromisoformat(start_time).replace(tzinfo=timezone.utc)
    
    # 1. Retrieve log entries from the training task execution window
    # We use a broad pod_pattern, but bound it precisely by the task start_time
    entries = list_log_entries(
        project_id=project_id,
        location=location,
        cluster_name=cluster_name,
        # Default target namespace for maxtext pods
        namespace="automation-testing", 
        pod_pattern=".*", 
        start_time=start_dt,
    )
    
    if not entries:
        print(f"Warning: No logs found in {cluster_name} since {start_dt}")
        return

    # 2. Buffer logic: Concatenate fragmented text payloads
    output_buffer = []
    for entry in entries:
        message = ""
        if isinstance(entry, logging_api.TextEntry):
            message = entry.payload
        elif isinstance(entry, logging_api.StructEntry):
            message = entry.payload.get("message", "")
        
        if message:
            output_buffer.append(message)
            
    full_text = "\n".join(output_buffer)
    
    # 3. Regex Verification Phase
    
    # (A) Check for non-UTF-8 replacement characters (\ufffd)
    if re.search(r'\ufffd', full_text):
        raise AirflowFailException(
            "Validation Failed: Found non-UTF-8 replacement characters (\ufffd) in the output. "
            "This indicates decoding errors or corrupted model inference."
        )
    
    # (B) Check for common silent errors (Loss becoming NaN, hidden Exceptions)
    # This prevents the job passing if the script ignored an exception or produced NaNs.
    if re.search(r'(?i)(Traceback \(most recent call last\):|\bNaN\b)', full_text):
        raise AirflowFailException(
            "Validation Failed: Found Tracebacks or NaNs in the output logs. "
            "The training or inference process might have crashed silently."
        )
        
    # (C) Check for Gibberish via severe repeating patterns 
    # Matches a sequence of 15+ characters repeating 5+ times consecutively (common in model collapse).
    if re.search(r'(.{15,})\1{5,}', full_text):
        raise AirflowFailException(
            "Validation Failed: Detected severe echoing/repeating patterns (Gibberish). "
            "The model output likely collapsed into a loop."
        )

    # (D) Generalized Structural Verification (Positive Assertion)
    # Instead of guessing what random gibberish looks like, the universally robust 
    # method for E2E testing is to assert that the model prints a specific required format.
    # Note: This requires the MaxText inference prompt to ask for this specific format
    # (e.g., "Always wrap your final answer in <RESULT>...</RESULT>").
    
    # Example 1: Tag matching
    # if not re.search(r'<RESULT>.*?</RESULT>', full_text, re.DOTALL):
    #     raise AirflowFailException(
    #         "Validation Failed: Model did not generate the expected <RESULT> tags. "
    #         "This indicates a loss of instruction-following capability or a gibberish collapse."
    #     )
    
    # Example 2: Strict JSON parsing (if Prompt demands JSON output)
    # try:
    #     # extract block between ```json ... ``` and parse
    #     pass
    # except json.JSONDecodeError:
    #      raise AirflowFailException("Validation Failed: Output is not valid JSON.")

    print("Semantic Inference output validation successful. No fatal patterns found.")
