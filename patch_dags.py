import re

files_to_patch = [
    "dags/tpu_observability/tpu_info_format_validation_dags.py",
    "dags/tpu_observability/tpu_info_format_validation_dags_exist.py"
]

for file_path in files_to_patch:
    with open(file_path, "r") as f:
        content = f.read()

    # 1. Add TaskGroupWithTimeout import
    if "TaskGroupWithTimeout" not in content:
        content = content.replace(
            "from airflow.utils.task_group import TaskGroup\n",
            "from airflow.utils.task_group import TaskGroup\nfrom dags.common.task_group_with_timeout import TaskGroupWithTimeout\n"
        )
        
    # 2. Replace top-level TaskGroup
    pattern = r"    with TaskGroup\(  # pylint: disable=unexpected-keyword-arg\n        group_id=f\"v\{config.tpu_version.value\}\"\n    \):"
    replacement = """    with TaskGroupWithTimeout(  # pylint: disable=unexpected-keyword-arg
        group_id=f"v{config.tpu_version.value}",
        timeout=DAGRUN_TIMEOUT,
    ):"""
    content = re.sub(pattern, replacement, content)
    
    # 3. Rewrite DAG task structure from create_node_pool down
    start_str = "      # Keyword arguments are generated dynamically at runtime (pylint does not\n      # know this signature).\n      with TaskGroup(  # pylint: disable=unexpected-keyword-arg\n          group_id=\"create_node_pool\"\n      ) as create_node_pool:"
    
    idx = content.find(start_str)
    if idx != -1:
        prefix = content[:idx]
        
        replacement = """      create_first_node_pool = node_pool.create.override(
          task_id="node_pool_1",
          retries=2,
      )(
          node_pool=cluster_info,
          node_pool_selector=selector,
      ).as_setup()

      create_second_node_pool = node_pool.create.override(
          task_id="node_pool_2",
          retries=2,
      )(
          node_pool=cluster_info_2,
          node_pool_selector=selector,
      ).as_setup()

      startup = jobset.create_jobset_startup_tasks(
          node_pool=cluster_info,
          jobset_config=jobset_config,
          jobset_name=jobset_name,
          node_pool_selector=selector,
          workload_type=Workload.JAX_TPU_BENCHMARK,
      )

      validate_format = validate_tpu_info_format.override(
          task_id="validate_tpu_info_format"
      )(
          info=cluster_info,
          tpu_config=config,
          pod_names=startup.running_pods,
      )

      clean_up_workload = jobset.end_workload.override(
          task_id="clean_up_workload", trigger_rule=TriggerRule.ALL_DONE
      )(
          node_pool=cluster_info,
          jobset_config=jobset_config,
          jobset_name=jobset_name,
      )

      cleanup_first_node_pool = node_pool.delete.override(
          task_id="cleanup_node_pool_1",
          trigger_rule=TriggerRule.ALL_DONE,
          retries=2,
      )(node_pool=cluster_info).as_teardown(setups=create_first_node_pool)

      cleanup_second_node_pool = node_pool.delete.override(
          task_id="cleanup_node_pool_2",
          trigger_rule=TriggerRule.ALL_DONE,
          retries=2,
      )(node_pool=cluster_info_2).as_teardown(setups=create_second_node_pool)

      chain(
          selector,
          jobset_name,
          [create_first_node_pool, create_second_node_pool],
          *startup.tasks,
          validate_format,
          clean_up_workload,
          [cleanup_first_node_pool, cleanup_second_node_pool],
      )
"""
        content = prefix + replacement

    with open(file_path, "w") as f:
        f.write(content)

