"""GCP Dataproc Cluster Optimizer

Manages Dataproc cluster lifecycle, auto-scaling policies,
and Spark job profiling to reduce cost and improve throughput.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from google.cloud import dataproc_v1
from google.cloud.dataproc_v1 import ClusterControllerClient
from google.protobuf import field_mask_pb2

logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    project_id: str
    region: str
    cluster_name: str
    master_machine_type: str = "n1-standard-4"
    worker_machine_type: str = "n1-standard-4"
    num_workers: int = 2
    preemptible_workers: int = 0
    min_instances: int = 2
    max_instances: int = 10
    scale_up_factor: float = 1.0
    scale_down_factor: float = 1.0
    scale_up_min_worker_fraction: float = 0.0
    graceful_decommission_timeout: str = "3600s"
    idle_delete_ttl: str = "1800s"


@dataclass
class JobProfile:
    job_id: str
    cluster_name: str
    duration_seconds: int
    yarn_memory_mb: int
    yarn_vcores: int
    input_bytes: int
    output_bytes: int
    shuffle_bytes: int
    stage_count: int
    task_count: int
    failed_tasks: int

    @property
    def efficiency_score(self) -> float:
        """Compute a 0-1 efficiency score for the job."""
        if self.task_count == 0:
            return 0.0
        success_rate = 1 - (self.failed_tasks / self.task_count)
        return round(success_rate, 4)


class DataprocClusterOptimizer:
    """Optimize Dataproc clusters via auto-scaling and job profiling."""

    def __init__(self, project_id: str, region: str):
        self.project_id = project_id
        self.region = region
        self.cluster_client = ClusterControllerClient(
            client_options={"api_endpoint": f"{region}-dataproc.googleapis.com:443"}
        )

    def create_optimized_cluster(self, config: ClusterConfig) -> dict:
        """Create a Dataproc cluster with auto-scaling and idle deletion."""
        cluster = {
            "project_id": config.project_id,
            "cluster_name": config.cluster_name,
            "config": {
                "master_config": {
                    "num_instances": 1,
                    "machine_type_uri": config.master_machine_type,
                    "disk_config": {"boot_disk_type": "pd-ssd", "boot_disk_size_gb": 100},
                },
                "worker_config": {
                    "num_instances": config.num_workers,
                    "machine_type_uri": config.worker_machine_type,
                    "disk_config": {"boot_disk_type": "pd-standard", "boot_disk_size_gb": 500},
                },
                "secondary_worker_config": {
                    "num_instances": config.preemptible_workers,
                    "is_preemptible": True,
                    "preemptibility": "PREEMPTIBLE",
                },
                "autoscaling_config": {
                    "policy_uri": self._get_autoscaling_policy_uri(
                        config.project_id, config.region, config.cluster_name
                    )
                },
                "lifecycle_config": {
                    "idle_delete_ttl": {"seconds": int(config.idle_delete_ttl.rstrip("s"))}
                },
                "software_config": {
                    "image_version": "2.1-debian11",
                    "properties": {
                        "spark:spark.sql.adaptive.enabled": "true",
                        "spark:spark.sql.adaptive.coalescePartitions.enabled": "true",
                        "spark:spark.dynamicAllocation.enabled": "true",
                        "yarn:yarn.nodemanager.resource.memory-mb": "6144",
                        "yarn:yarn.nodemanager.resource.cpu-vcores": "4",
                    },
                },
            },
        }
        operation = self.cluster_client.create_cluster(
            request={
                "project_id": config.project_id,
                "region": config.region,
                "cluster": cluster,
            }
        )
        logger.info("Creating cluster: %s", config.cluster_name)
        result = operation.result()
        logger.info("Cluster created: %s", result.cluster_name)
        return result

    def create_autoscaling_policy(
        self,
        policy_id: str,
        min_instances: int,
        max_instances: int,
        scale_up_factor: float = 1.0,
        scale_down_factor: float = 1.0,
    ) -> dict:
        """Create an auto-scaling policy for a Dataproc cluster."""
        from google.cloud import dataproc_v1 as dataproc

        policy_client = dataproc.AutoscalingPolicyServiceClient(
            client_options={"api_endpoint": f"{self.region}-dataproc.googleapis.com:443"}
        )
        policy = {
            "id": policy_id,
            "basic_algorithm": {
                "yarn_config": {
                    "scale_up_factor": scale_up_factor,
                    "scale_down_factor": scale_down_factor,
                    "scale_up_min_worker_fraction": 0.0,
                    "scale_down_min_worker_fraction": 0.0,
                    "graceful_decommission_timeout": {"seconds": 3600},
                },
                "cooldown_period": {"seconds": 120},
            },
            "worker_config": {
                "min_instances": min_instances,
                "max_instances": max_instances,
                "weight": 1,
            },
            "secondary_worker_config": {
                "min_instances": 0,
                "max_instances": max_instances * 2,
                "weight": 1,
            },
        }
        parent = f"projects/{self.project_id}/regions/{self.region}"
        result = policy_client.create_autoscaling_policy(
            request={"parent": parent, "policy": policy}
        )
        logger.info("Created autoscaling policy: %s", policy_id)
        return result

    def _get_autoscaling_policy_uri(self, project: str, region: str, policy_id: str) -> str:
        return f"projects/{project}/regions/{region}/autoscalingPolicies/{policy_id}"

    def resize_cluster(
        self,
        cluster_name: str,
        num_workers: int,
        num_preemptible: int = 0,
    ) -> None:
        """Manually resize a running cluster."""
        cluster = self.cluster_client.get_cluster(
            request={
                "project_id": self.project_id,
                "region": self.region,
                "cluster_name": cluster_name,
            }
        )
        cluster.config.worker_config.num_instances = num_workers
        cluster.config.secondary_worker_config.num_instances = num_preemptible

        update_mask = field_mask_pb2.FieldMask(
            paths=[
                "config.worker_config.num_instances",
                "config.secondary_worker_config.num_instances",
            ]
        )
        operation = self.cluster_client.update_cluster(
            request={
                "project_id": self.project_id,
                "region": self.region,
                "cluster_name": cluster_name,
                "cluster": cluster,
                "update_mask": update_mask,
            }
        )
        operation.result()
        logger.info(
            "Resized %s to %d workers + %d preemptible",
            cluster_name, num_workers, num_preemptible
        )

    def delete_idle_clusters(self, max_idle_minutes: int = 30) -> List[str]:
        """Find and delete clusters idle for longer than max_idle_minutes."""
        deleted = []
        clusters = self.cluster_client.list_clusters(
            request={"project_id": self.project_id, "region": self.region}
        )
        for cluster in clusters:
            if cluster.status.state.name == "RUNNING":
                # Check idle time via metrics
                idle_secs = cluster.metrics.hdfs_metrics.get("dfs-used", 0)
                # In real implementation, check YARN metrics for idle time
                logger.info("Cluster %s state: RUNNING", cluster.cluster_name)
        return deleted

    def generate_cost_report(self, job_profiles: List[JobProfile]) -> str:
        """Generate a cost and efficiency report from job profiles."""
        lines = ["# Dataproc Job Efficiency Report\n"]
        lines.append("| Job ID | Duration (s) | Efficiency | Failed Tasks | Shuffle GB |")
        lines.append("|--------|--------------|------------|--------------|------------|")
        for jp in job_profiles:
            lines.append(
                f"| {jp.job_id} | {jp.duration_seconds} | {jp.efficiency_score:.2%} "
                f"| {jp.failed_tasks} | {jp.shuffle_bytes / 1e9:.2f} |"
            )
        avg_eff = sum(j.efficiency_score for j in job_profiles) / max(len(job_profiles), 1)
        lines.append(f"\n**Average Efficiency: {avg_eff:.2%}**")
        return "\n".join(lines)


if __name__ == "__main__":
    config = ClusterConfig(
        project_id="my-gcp-project",
        region="us-central1",
        cluster_name="optimized-etl-cluster",
        master_machine_type="n1-standard-8",
        worker_machine_type="n1-standard-4",
        num_workers=4,
        preemptible_workers=8,
        min_instances=2,
        max_instances=20,
    )
    optimizer = DataprocClusterOptimizer(
        project_id=config.project_id,
        region=config.region,
    )
    print(f"Optimizer initialized for project: {config.project_id}")
