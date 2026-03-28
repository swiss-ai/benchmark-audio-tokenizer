import logging
import os
from pathlib import Path
import json

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Print config only on rank 0
    if(int(os.environ.get("RANK", 0)) == 0):
        logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Convert Hydra DictConfig to a plain dict
    pipeline_cfg = OmegaConf.to_container(cfg, resolve=True)

    # Allow comma separated SHAR dir for preprocessing
    shar_dir = pipeline_cfg["shar_dir"]
    if isinstance(shar_dir, str) and "," in shar_dir:
        pipeline_cfg["shar_dir"] = shar_dir.split(",")
    
    from .pipeline import run_pipeline
    result = run_pipeline(pipeline_cfg)

    rank = int(os.environ.get("RANK", 0))
    output_dir = Path(pipeline_cfg["output_dir"])
    summary_path = output_dir / f"summary_{rank:04d}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)

    #----Implement final summary.json (One Rank has to wait for other)--------
    # if rank == 0:
    #     final_result = {
    #             "samples_processed": 0,
    #             "samples_skipped": 0,
    #             "errors": 0,
    #             "total_audio_seconds": 0,
    #             "elapsed_time": 0,
    #             "throughput": 0,
    #             }
    #     summary_files = list(output_dir.glob("summary_*.json"))
    #     num_files = len(summary_files)

    #     if num_files > 0:
    #         for file in summary_files:
    #             with open(file, "r") as f:
    #                 data = json.load(f) 
                    
    #                 final_result["samples_processed"] += data.get("samples_processed", 0)
    #                 final_result["samples_skipped"] += data.get("samples_skipped", 0)
    #                 final_result["errors"] += data.get("errors", 0)
    #                 final_result["total_audio_seconds"] += data.get("total_audio_seconds", 0)
                    
    #                 final_result["elapsed_time"] += data.get("elapsed_time", 0)
    #                 final_result["throughput"] += data.get("throughput", 0)
    #         final_result["elapsed_time"] /= num_files

    #     final_summary_path = output_dir / "summary.json" 
    #     with open(final_summary_path, "w") as f:
    #         json.dump(final_result, f, indent=2)

    logger.info(f"Summary written to {summary_path}")

if __name__ == "__main__":
    main()