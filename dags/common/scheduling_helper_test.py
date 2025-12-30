# This test file should be run from the project root directory using:
# python -m unittest discover dags/common -p "scheduling_helper_test.py"
#
# Other methods that might work:
# 1. Specific test: python -m unittest dags.common.scheduling_helper_test.TestSampleSchedulingHelper.test_schedule_success

from absl.testing import absltest
from dags.common.scheduling_helper import SchedulingHelper
from dags.common.scheduling_helper import Dag
from unittest.mock import patch
import ast
import os
import datetime as dt
from dags.common.vm_resource import XpkClusters


def ExtractDagIds(file_path: str) -> list[str]:
  """Extract all DAG IDs from a Python file."""
  with open(file_path, "r", encoding="utf-8") as f:
    tree = ast.parse(f.read(), filename=file_path)

  dag_ids = []
  constants = CollectConstants(tree)

  # Module
  # └─ body[0]: With
  #    ├─ items[0]: withitem
  #    │  └─ context_expr: Call
  #    │     ├─ func: Attribute
  #    │     │  ├─ value: Name(id='models')
  #    │     │  └─ attr: 'DAG'
  #    │     ├─ args: []
  #    │     └─ keywords: [
  #    │        └─ keyword
  #    │           ├─ arg: 'dag_id'
  #    │           └─ value: Name(varialbe name) or String
  #    └─ body: []

  for node in ast.walk(tree):
    if isinstance(node, ast.With):
      for item in node.items:
        code_node = item.context_expr
        if isinstance(code_node, ast.Call):
          func = code_node.func
          if (isinstance(func, ast.Name) and func.id == "DAG") or (
              isinstance(func, ast.Attribute) and func.attr == "DAG"
          ):
            for kw in code_node.keywords:
              if kw.arg == "dag_id":
                if isinstance(kw.value, ast.Constant):
                  dag_ids.append(str(kw.value.value))
                elif isinstance(kw.value, ast.Name):
                  dag_ids.append(constants.get(kw.value.id))

  return dag_ids


def CollectConstants(tree) -> dict[str, str]:
  """Collect all string constants from the AST tree."""
  constants = {}
  for node in tree.body:
    if isinstance(node, ast.Assign):
      if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        if isinstance(node.value, ast.Constant) and isinstance(
            node.value.value, str
        ):
          constants[node.targets[0].id] = node.value.value
  return constants


def GetAllDag(folder_path: str) -> list[str]:
  """Get all DAG id from all Python files under target folder."""

  dag_ids = []
  for filename in os.listdir(folder_path):
    if filename.endswith(".py"):
      file_path = os.path.join(folder_path, filename)
      dag_ids.extend(ExtractDagIds(file_path))

  return dag_ids


class TestSampleSchedulingHelper(absltest.TestCase):
  """
  Test cases for the SchedulingHelper class.
  """

  def test_schedule_success(self):
    """
    Test scheduling helper ArrangeScheduleTime with success result.
    """

    schedule = SchedulingHelper.ArrangeScheduleTime(
        XpkClusters.TPU_V5P_128_CLUSTER,
        "maxtext_mtc_resume_from_gcs",
    )

    registered_cluster = SchedulingHelper.registry[
        XpkClusters.TPU_V5P_128_CLUSTER
    ]
    default_margin = SchedulingHelper.DEFAULT_MARGIN
    anchor = SchedulingHelper.DEFAULT_ANCHOR
    offset = dt.timedelta(0)
    ans = ""
    for dag in registered_cluster:
      if "maxtext_mtc_resume_from_gcs" == dag.dag_id:
        accumulate_time = anchor + offset
        ans = f"{accumulate_time.minute} {accumulate_time.hour} * * *"
        break

      offset += default_margin + dag.dag_run_timeout
    self.assertEqual(schedule, ans)

  def test_all_dags_scheduled(self):
    """
    Test all DAGs in the registry are scheduled.
    """
    is_all_scheduled = True
    dags_list = GetAllDag("dags/orbax")
    for dag_id in dags_list:
      try:
        SchedulingHelper.ArrangeScheduleTime(
            XpkClusters.TPU_V5P_128_CLUSTER,
            dag_id,
        )
      except ValueError:
        is_all_scheduled = False
        break
    self.assertTrue(is_all_scheduled)

  def test_unregistered_cluster(self):
    """
    Test scheduling helper with nonexistent cluster.
    """
    with self.assertRaises(ValueError) as ctx:
      SchedulingHelper.ArrangeScheduleTime(
          XpkClusters.TPU_V5E_256_CLUSTER,
          "maxtext_mtc_resume_from_gcs",
      )

    self.assertIn(
        f"{XpkClusters.TPU_V5E_256_CLUSTER.name} is not found in the registry",
        str(ctx.exception),
    )

  def test_unregistered_dag_id(self):
    """
    Test scheduling helper with nonexistent dag_id.
    """
    with self.assertRaises(ValueError) as ctx:
      SchedulingHelper.ArrangeScheduleTime(
          XpkClusters.TPU_V5P_128_CLUSTER,
          "nonexistent_dag_id",
      )

    self.assertIn(
        "nonexistent_dag_id is not found in the registry",
        str(ctx.exception),
    )

  def test_overtime_dag(self):
    """
    Test ArrangeScheduleTime raises ValueError if total DAG schedule
    exceeds 24 hours.
    """
    fake_dags = [
        Dag(f"fake_dag_{i}", dt.timedelta(minutes=145)) for i in range(10)
    ]

    new_registry = SchedulingHelper.registry.copy()
    cluster = XpkClusters.TPU_V5P_128_CLUSTER
    new_registry[cluster] = new_registry.get(cluster, []) + fake_dags

    with patch.object(SchedulingHelper, "registry", new_registry):
      with self.assertRaises(ValueError) as ctx:
        SchedulingHelper.ArrangeScheduleTime(
            XpkClusters.TPU_V5P_128_CLUSTER,
            "fake_dag_9",
        )

    self.assertIn("Schedule exceeds 24 hours window", str(ctx.exception))


if __name__ == "__main__":
  absltest.main()
