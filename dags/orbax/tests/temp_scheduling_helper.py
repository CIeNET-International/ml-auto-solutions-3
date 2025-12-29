"""dags.orbax.tests.scheduling_helper 的 Docstring"""

from dags.common.scheduling_helper import SchedulingHelper
from dags.common.vm_resource import XpkClusters
from dags.common.scheduling_helper import Dag
from dags.common.scheduling_helper import XpkClusterConfig
import datetime as dt

class TempSchedulingHelper(SchedulingHelper):
  """
  A temporary SchedulingHelper class containing errors for testing purposes.
  """

  registry: dict[XpkClusterConfig, list[Dag]] = {
      XpkClusters.TPU_V5P_128_CLUSTER: [
          Dag("test_dag_1", dt.timedelta(minutes=360)),
          Dag("test_dag_2", dt.timedelta(minutes=360)),
          Dag("test_dag_3", dt.timedelta(minutes=360)),
          Dag("test_dag_4", dt.timedelta(minutes=360)),
          Dag("test_dag_5", dt.timedelta(minutes=360)),
      ],
  }


