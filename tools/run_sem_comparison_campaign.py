"""Run the locked A3-vs-SEM-RRT*-adapted common-objective campaign.

Dataset generation/qualification and build/correctness-test gates happen before this
command.  The cumulative campaign clock covers planner collection, independent
offline validation, analysis, and report finalization, with a hard 7,200 s cap.
"""
import argparse
from pathlib import Path

from experiment_common import ROOT,default_worker,load_config,read_json
from run_advanced_campaign import run_campaign,verify_dataset
from run_experiments import (TRIAL_ORDER_SEM_COMPARISON,audit_sem_comparison_plan,
                             plan)


def require_explicit_allow(allowed):
    if not allowed:
        raise ValueError("SEM comparison requires explicit --allow-sem-comparison")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,
                        default=ROOT/"configs/a3_vs_sem_rrtstar_common_v1.yaml")
    parser.add_argument("--map-manifest",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--report-output",type=Path,required=True)
    parser.add_argument("--worker",type=Path,default=default_worker())
    parser.add_argument("--allow-sem-comparison",action="store_true")
    parser.add_argument("--resume",action="store_true",
                        help="continue only with exact identity and cumulative persisted campaign time")
    parser.add_argument("--dry-run",action="store_true",
                        help="validate and print the locked plan without creating campaign output")
    args=parser.parse_args();require_explicit_allow(args.allow_sem_comparison)
    if args.dry_run and args.resume:
        parser.error("--dry-run and --resume are mutually exclusive")
    if args.dry_run:
        config=load_config(args.config,"sem_comparison");maps=read_json(args.map_manifest)
        verify_dataset(config,maps)
        trials=plan(config,"sem_comparison",maps,order_policy=TRIAL_ORDER_SEM_COMPARISON)
        print({"planned_trials":len(trials),
               "schedule":audit_sem_comparison_plan(trials,config["profiles"]["sem_comparison"]),
               "campaign_clock":"NOT_STARTED_DRY_RUN"})
        return 0
    return run_campaign(args.config,args.map_manifest,args.output,args.report_output,args.worker,
                        resume=args.resume,campaign_kind="sem_comparison")


if __name__=="__main__":raise SystemExit(main())
