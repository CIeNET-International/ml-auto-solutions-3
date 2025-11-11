"""Utility functions for running subprocess commands."""

import logging
import subprocess

from airflow.exceptions import AirflowFailException


def run_exec(
    cmd: str,
    env: dict[str, str] | None = None,
    log_command=False,
    log_output: bool = True,
) -> str:
  """Executes a shell command and logs its output."""
  if log_command:
    logging.info("[subprocess] executing command:\n %s\n", cmd)

  res = subprocess.run(
      cmd,
      env=env,
      shell=True,
      check=False,
      capture_output=True,
      text=True,
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
