"""Utility functions for running subprocess commands."""

import logging
import subprocess

from airflow.exceptions import AirflowFailException


def run_exec(
    cmd: str,
    env: dict[str, str] | None = None,
    log_command: bool = False,
    log_output: bool = True,
) -> str:
  """Executes a shell command and logs its output."""
  if log_command:
    logging.info("[subprocess] executing command:\n %s\n", cmd)

  res = subprocess.run(
      cmd,  # The command string to execute (e.g., 'ls -l | grep file')
      # Optional: Environment variables to use for the command
      # (overrides parent environment)
      env=env,
      shell=True,  # REQUIRED for shell features (e.g., pipes, redirects).
      # Optional: Do NOT raise CalledProcessError if the command returns a
      # non-zero exit code. We handle errors manually.
      check=False,
      # Optional: Capture stdout and stderr in the result object.
      capture_output=True,
      text=True,  # Optional: Decode stdout/stderr output as text strings
      # (using the default system encoding).
  )

  if res.returncode != 0:
    logging.info("[subprocess] stderr: %s", res.stderr)
    raise AirflowFailException(
        "Caught an error while executing a command. stderr Message:"
        f" {res.stderr}"
    )

  if log_output:
    logging.info("[subprocess] stdout: %s", res.stdout)

  return res.stdout
