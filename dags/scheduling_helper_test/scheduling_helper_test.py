"""Test script for SchedulingHelper."""

from dags.scheduling_helper_test.scheduling_helper import SchedulingHelper
from dags.common.vm_resource import XpkClusters


if __name__ == "__main__":
  for i in range(1, 13):
    try:
      test_case = SchedulingHelper.ArrangeScheduleTime(
          XpkClusters.TPU_V5P_128_CLUSTER, f"test{i}"
      )
      print(f"Test case {i} schedule: {test_case}")
    except ValueError as e:
      print(f"Test case {i} raised an error: {e}")
