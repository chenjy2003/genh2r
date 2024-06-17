import os
import ray
import numpy as np
from numpy.typing import NDArray
from omegaconf import OmegaConf
import time
import code

from env.benchmark import get_scene_ids
from env.status_checker import EpisodeStatus
from evaluate_config import EvaluateConfig
from policy_runner import result_dtype, Distributer, PolicyRunner, get_policy_runner_remote

def print_results(cfg: EvaluateConfig, results: NDArray[result_dtype]):
    num_scenes = results.shape[0]
    success_cnt = (results["status"] == EpisodeStatus.SUCCESS).sum()
    contact_cnt = (results["status"] == EpisodeStatus.FAILURE_HUMAN_CONTACT).sum()
    drop_cnt    = (results["status"] == EpisodeStatus.FAILURE_OBJECT_DROP).sum()
    timeout_cnt = (results["status"] == EpisodeStatus.FAILURE_TIMEOUT).sum()
    print(f"success rate: {success_cnt}/{num_scenes}={success_cnt/num_scenes}")
    print(f"contact rate: {contact_cnt}/{num_scenes}={contact_cnt/num_scenes}")
    print(f"   drop rate: {drop_cnt}/{num_scenes}={drop_cnt/num_scenes}")
    print(f"timeout rate: {timeout_cnt}/{num_scenes}={timeout_cnt/num_scenes}")
    success_mask = results["status"] == EpisodeStatus.SUCCESS
    # average_done_frame = results["done_frame"].mean()
    average_success_done_frame = results["done_frame"][success_mask].mean()
    average_success_reached_frame = results["reached_frame"][success_mask].mean()
    average_success_num_steps = results["num_steps"][success_mask].mean()
    average_success = ((cfg.env.max_frames-results["done_frame"][success_mask]+1)/cfg.env.max_frames).sum()/num_scenes
    # print(f"average done frame        : {average_done_frame}")
    print(f"average success done frame   : {average_success_done_frame}")
    print(f"average success reached frame: {average_success_reached_frame}")
    print(f"average success num steps    : {average_success_num_steps}")
    print(f"average success              : {average_success}")
    if cfg.print_failure_ids:
        print(f"contact indices: {sorted(results['scene_id'][results['status'] == EpisodeStatus.FAILURE_HUMAN_CONTACT])}")
        print(f"   drop indices: {sorted(results['scene_id'][results['status'] == EpisodeStatus.FAILURE_OBJECT_DROP])}")
        print(f"timeout indices: {sorted(results['scene_id'][results['status'] == EpisodeStatus.FAILURE_TIMEOUT])}")

def evaluate(cfg: EvaluateConfig):
    start_time = time.time()
    if cfg.scene_ids is None:
        scene_ids = get_scene_ids(cfg.setup, cfg.split, cfg.start_object_idx, cfg.end_object_idx, cfg.start_traj_idx, cfg.end_traj_idx)
    else:
        scene_ids = cfg.scene_ids
    
    ray.init()
    distributer = Distributer.remote(scene_ids)
    if not cfg.use_ray:
        policy_runner = PolicyRunner(cfg, distributer)
        results = policy_runner.work()
    else:
        num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        num_runners_per_gpu = (cfg.num_runners-1)//num_gpus+1
        PolicyRunnerRemote = get_policy_runner_remote(1/num_runners_per_gpu)
        policy_runners = [PolicyRunnerRemote.remote(cfg, distributer) for _ in range(cfg.num_runners)]
        results = ray.get([policy_runner.work.remote() for policy_runner in policy_runners])
        results: NDArray[result_dtype] = np.concatenate(results)
    results = np.sort(results, order="scene_id")
    print_results(cfg, results)
    if cfg.demo_dir is not None:
        # results.tofile(os.path.join(cfg.demo_dir, "results.bin"))
        if cfg.start_object_idx is not None or cfg.end_object_idx is not None or cfg.start_traj_idx is not None or cfg.end_traj_idx is not None:
            np.save(os.path.join(cfg.demo_dir, f"results_{cfg.start_object_idx}_{cfg.end_object_idx}_{cfg.start_traj_idx}_{cfg.end_traj_idx}.npy"), results)
        else:
            np.save(os.path.join(cfg.demo_dir, f"results.npy"), results)
    print(f"evaluting uses {time.time()-start_time} seconds")
    # code.interact(local=dict(globals(), **locals()))
    ray.shutdown()

def main():
    default_cfg = OmegaConf.structured(EvaluateConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg: EvaluateConfig = OmegaConf.to_object(OmegaConf.merge(default_cfg, cli_cfg))
    if cfg.start_seed is not None and cfg.end_seed is not None and cfg.step_seed is not None:
        root_demo_dir = cfg.demo_dir
        for seed in range(cfg.start_seed, cfg.end_seed, cfg.step_seed):
            print(f"evaluating for seed {seed}")
            cfg.seed = seed
            cfg.demo_dir = os.path.join(root_demo_dir, str(seed))
            evaluate(cfg)
    else:
        evaluate(cfg)

if __name__ == "__main__":
    main()

"""
CUDA_VISIBLE_DEVICES=0 python -m evaluate use_ray=False env.panda.IK_solver=PyKDL setup=s0 split=train policy=offline offline.demo_dir=/data2/haoran/HRI/expert_demo/OMG/s0/know_dest_smooth_0.08 offline.demo_source=handoversim env.visualize=True env.verbose=True

CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_DEDUP_LOGS=0 python -m evaluate env.panda.IK_solver=PyKDL setup=s0 split=train policy=offline offline.demo_dir=/data2/haoran/HRI/expert_demo/OMG/s0/know_dest_smooth_0.08 offline.demo_source=handoversim
success rate: 567/720=0.7875
contact rate: 37/720=0.05138888888888889
   drop rate: 50/720=0.06944444444444445
timeout rate: 66/720=0.09166666666666666
average done frame        : 6500.272222222222
average success done frame: 6042.241622574956
average success num steps : 44.16225749559083
average success           : 0.42154017094017093
evaluting uses 266.8944182395935 seconds
# original: success 546/720=0.7583333333333333

python -m evaluate env.panda.IK_solver PyKDL evaluate.SCENE_IDS "[951, 996, 953, 917]" evaluate.use_ray False policy.name offline policy.offline.demo_dir /data2/haoran/HRI/expert_demo/OMG/s0/know_dest_smooth_0.08 policy.offline.demo_structure flat env.visualize True env.verbose True
"""