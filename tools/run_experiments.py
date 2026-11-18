import argparse
import datetime as dt
from pathlib import Path
import random
import re
import subprocess
import time
import zipfile
import psutil

from experiment_common import ROOT,code_hash,default_worker,digest,dump,environment,file_hash,load_config,read_json,resolved_planner_options,source_files,worker_env

TRIAL_ORDER_INDIVIDUAL="individual_trial_shuffle_v1"
TRIAL_ORDER_PAIRED="paired_problem_slots_v1"
TRIAL_ORDER_LATIN="latin_square_problem_slots_v1"
TRIAL_ORDER_SEM_COMPARISON="paired_map_parity_seed_v1"
TRIAL_ORDER_COMBINED="latin_square_five_planners_v1"
PAIRED_PLANNERS=("bitstar_a0","bitstar_a3","informed_rrtstar","rrtstar")
PAIRED_PLANNER_CYCLE=(
    ("bitstar_a0","bitstar_a3","informed_rrtstar","rrtstar"),
    ("bitstar_a3","bitstar_a0","rrtstar","informed_rrtstar"),
    ("informed_rrtstar","rrtstar","bitstar_a0","bitstar_a3"),
    ("rrtstar","informed_rrtstar","bitstar_a3","bitstar_a0"),
)
ADVANCED_PLANNERS=("bitstar_a3","abitstar","aitstar","lazyprmstar_hauser2015")
ADVANCED_LATIN_SQUARE=tuple(
    ADVANCED_PLANNERS[offset:]+ADVANCED_PLANNERS[:offset]
    for offset in range(len(ADVANCED_PLANNERS)))
SEM_COMPARISON_PLANNERS=("bitstar_a3","sem_rrtstar_common_v1")
COMBINED_PLANNERS=("bitstar_a3","abitstar","aitstar","lazyprmstar_hauser2015",
                   "sem_rrtstar_common_v1")
COMBINED_LATIN_SQUARE=tuple(
    COMBINED_PLANNERS[offset:]+COMBINED_PLANNERS[:offset]
    for offset in range(len(COMBINED_PLANNERS)))

def utc():return dt.datetime.now(dt.timezone.utc).isoformat()

def bitstar_log_counters(text):
    # Source: BITstar::endSuccessMessage/endFailureMessage. Require the final
    # message, so an interrupted process cannot report an earlier event as final.
    final=[line for line in text.splitlines() if "Finished with a solution" in line or "Did not find an exact solution" in line]
    rewires=re.findall(r"perform (\d+) rewirings",final[-1]) if final else []
    accepted=re.findall(r"(?:found from|solution from) (\d+) samples",final[-1]) if final else []
    return {"rewires":int(rewires[-1]) if rewires else None,"samples_accepted":int(accepted[-1]) if accepted else None,
        "rewires_measurement":"native_BITstar_final_log_numRewirings" if rewires else "unavailable_no_native_final_message",
        "samples_accepted_measurement":"ImplicitGraph_numSamples_after_sampler_and_state_validity_excludes_explicit_goal_vertices" if accepted else "unavailable_no_native_final_message"}


def _balanced_problem_slot_subset(slots,settings,limit,rng):
    """Select a deterministic limited subset without looking at planner outcomes.

    Quotas are balanced across configured families. Within a family, selection
    round-robins over shuffled maps, so a 10-slot/family G5 subset covers all
    eight maps before repeating any map.
    """
    if not isinstance(limit,int) or limit<=0 or limit>len(slots):
        raise ValueError("invalid max_problem_slots")
    families=list(settings["families"])
    base,extra=divmod(limit,len(families))
    selected=[]
    for family_index,family in enumerate(families):
        quota=base+(1 if family_index<extra else 0)
        candidates=[slot for slot in slots if slot[1][0]["family"]==family]
        by_map={}
        for slot in candidates:by_map.setdefault(slot[1][0]["map_id"],[]).append(slot)
        map_ids=sorted(by_map)
        rng.shuffle(map_ids)
        for values in by_map.values():rng.shuffle(values)
        if quota>sum(map(len,by_map.values())):
            raise ValueError("family lacks enough problem slots")
        for offset in range(quota):
            map_id=map_ids[offset%len(map_ids)]
            selected.append(by_map[map_id].pop())
    rng.shuffle(selected)
    return selected


def audit_paired_plan(trials,block_size=None):
    if len(trials)%len(PAIRED_PLANNERS):
        raise ValueError("paired plan ends inside a problem slot")
    if block_size is not None and (not isinstance(block_size,int) or block_size<=0 or block_size%len(PAIRED_PLANNERS)):
        raise ValueError("paired validation block must contain whole problem slots")
    slots=[];a0_first=a3_first=0
    for offset in range(0,len(trials),len(PAIRED_PLANNERS)):
        group=trials[offset:offset+len(PAIRED_PLANNERS)]
        keys={(entry["map_id"],entry["map_hash"],entry["query_id"],entry["query_hash"],entry["planner_seed"],
              tuple(entry["start"]),tuple(entry["goal"])) for entry in group}
        if len(keys)!=1 or len({entry["problem_slot_id"] for entry in group})!=1:
            raise ValueError("paired problem slot identity mismatch")
        if {entry["planner_id"] for entry in group}!=set(PAIRED_PLANNERS):
            raise ValueError("paired problem slot planner set mismatch")
        positions={entry["planner_id"]:index for index,entry in enumerate(group)}
        if abs(positions["bitstar_a0"]-positions["bitstar_a3"])!=1:
            raise ValueError("A0/A3 are not adjacent")
        if [entry["problem_slot_order"] for entry in group]!=[offset//len(PAIRED_PLANNERS)]*len(PAIRED_PLANNERS):
            raise ValueError("problem slot order mismatch")
        if [entry["within_slot_order"] for entry in group]!=list(range(len(PAIRED_PLANNERS))):
            raise ValueError("within-slot order mismatch")
        if positions["bitstar_a0"]<positions["bitstar_a3"]:a0_first+=1
        else:a3_first+=1
        slots.append(group[0]["problem_slot_id"])
    if len(slots)!=len(set(slots)):
        raise ValueError("duplicate problem slot")
    if [entry["shuffle_order"] for entry in trials]!=list(range(len(trials))):
        raise ValueError("non-contiguous shuffle order")
    blocks=[]
    if block_size is not None:
        blocks=[len(trials[start:start+block_size])//len(PAIRED_PLANNERS) for start in range(0,len(trials),block_size)]
    return {"policy":TRIAL_ORDER_PAIRED,"planned_trials":len(trials),"problem_slots":len(slots),
            "trials_per_problem_slot":len(PAIRED_PLANNERS),"a0_first":a0_first,"a3_first":a3_first,
            "validation_block_size":block_size,"problem_slots_per_block":blocks}


def audit_latin_plan(trials):
    """Verify complete four-planner slots and Latin-square position balance."""
    size=len(ADVANCED_PLANNERS)
    if len(trials)%size:
        raise ValueError("Latin-square plan ends inside a planner block")
    positions={planner:[0]*size for planner in ADVANCED_PLANNERS}
    slots=[]
    for offset in range(0,len(trials),size):
        group=trials[offset:offset+size]
        keys={(entry["map_id"],entry["map_hash"],entry["query_id"],entry["query_hash"],entry["planner_seed"],
              tuple(entry["start"]),tuple(entry["goal"])) for entry in group}
        if len(keys)!=1 or len({entry.get("problem_slot_id") for entry in group})!=1:
            raise ValueError("Latin-square problem slot identity mismatch")
        if tuple(entry["planner_id"] for entry in group)!=ADVANCED_LATIN_SQUARE[(offset//size)%size]:
            raise ValueError("Latin-square row mismatch")
        if [entry.get("problem_slot_order") for entry in group]!=[offset//size]*size:
            raise ValueError("Latin-square problem slot order mismatch")
        if [entry.get("within_slot_order") for entry in group]!=list(range(size)):
            raise ValueError("Latin-square within-slot order mismatch")
        for index,entry in enumerate(group):positions[entry["planner_id"]][index]+=1
        slots.append(group[0]["problem_slot_id"])
    if len(slots)!=len(set(slots)):
        raise ValueError("duplicate Latin-square problem slot")
    if [entry.get("shuffle_order") for entry in trials]!=list(range(len(trials))):
        raise ValueError("non-contiguous Latin-square shuffle order")
    target=len(slots)//size
    if len(slots)%size or any(count!=target for counts in positions.values() for count in counts):
        raise ValueError("Latin-square position counts are not exactly balanced")
    return {"policy":TRIAL_ORDER_LATIN,"planned_trials":len(trials),"problem_slots":len(slots),
            "trials_per_problem_slot":size,"latin_square":list(map(list,ADVANCED_LATIN_SQUARE)),
            "position_counts":positions,"complete_planner_blocks":True}


def _sem_comparison_order(settings,map_seed,planner_seed):
    try:
        map_index=settings["map_seeds"].index(map_seed)
        seed_index=settings["planner_seeds"].index(planner_seed)
    except ValueError as error:
        raise ValueError("SEM comparison slot contains an unregistered map/planner seed") from error
    a3_first=(map_index+seed_index)%2==0
    return SEM_COMPARISON_PLANNERS if a3_first else tuple(reversed(SEM_COMPARISON_PLANNERS))


def audit_sem_comparison_plan(trials,settings):
    """Verify complete two-planner pairs and the predeclared parity balance."""
    size=len(SEM_COMPARISON_PLANNERS)
    if len(trials)%size:
        raise ValueError("SEM comparison plan ends inside a planner pair")
    first={planner:0 for planner in SEM_COMPARISON_PLANNERS}
    family_first={family:{planner:0 for planner in SEM_COMPARISON_PLANNERS}
                  for family in settings["families"]}
    # JSON object keys are strings. Keep the in-memory schedule JSON-native so
    # strict resume equality survives dump/read round trips.
    seed_first={str(seed):{planner:0 for planner in SEM_COMPARISON_PLANNERS}
                for seed in settings["planner_seeds"]}
    map_first={};slots=[]
    for offset in range(0,len(trials),size):
        group=trials[offset:offset+size]
        keys={(entry["map_id"],entry["map_hash"],entry["query_id"],entry["query_hash"],
              entry["planner_seed"],tuple(entry["start"]),tuple(entry["goal"])) for entry in group}
        if len(keys)!=1 or len({entry.get("problem_slot_id") for entry in group})!=1:
            raise ValueError("SEM comparison problem slot identity mismatch")
        expected=_sem_comparison_order(settings,group[0]["map_seed"],group[0]["planner_seed"])
        if tuple(entry["planner_id"] for entry in group)!=expected:
            raise ValueError("SEM comparison within-pair parity order mismatch")
        if [entry.get("problem_slot_order") for entry in group]!=[offset//size]*size:
            raise ValueError("SEM comparison problem slot order mismatch")
        if [entry.get("within_slot_order") for entry in group]!=list(range(size)):
            raise ValueError("SEM comparison within-slot order mismatch")
        leader=group[0];planner=leader["planner_id"]
        first[planner]+=1;family_first[leader["family"]][planner]+=1
        seed_first[str(leader["planner_seed"])][planner]+=1
        by_map=map_first.setdefault(leader["map_id"],{name:0 for name in SEM_COMPARISON_PLANNERS})
        by_map[planner]+=1;slots.append(leader["problem_slot_id"])
    if len(slots)!=len(set(slots)):
        raise ValueError("duplicate SEM comparison problem slot")
    if [entry.get("shuffle_order") for entry in trials]!=list(range(len(trials))):
        raise ValueError("non-contiguous SEM comparison shuffle order")
    if first!={planner:len(slots)//2 for planner in SEM_COMPARISON_PLANNERS}:
        raise ValueError("SEM comparison global first-position counts are not balanced")
    family_target=len(settings["map_seeds"])*len(settings["planner_seeds"])//2
    if any(count!=family_target for values in family_first.values() for count in values.values()):
        raise ValueError("SEM comparison family first-position counts are not balanced")
    seed_target=len(settings["families"])*len(settings["map_seeds"])//2
    if any(count!=seed_target for values in seed_first.values() for count in values.values()):
        raise ValueError("SEM comparison seed first-position counts are not balanced")
    if any(sorted(values.values())!=[1,2] for values in map_first.values()):
        raise ValueError("SEM comparison per-map order must alternate 2/1 across three seed slots")
    return {"policy":TRIAL_ORDER_SEM_COMPARISON,"planned_trials":len(trials),
            "problem_slots":len(slots),"trials_per_problem_slot":size,"first_position_counts":first,
            "family_first_position_counts":family_first,"seed_first_position_counts":seed_first,
            "complete_planner_blocks":True}


def _combined_row_index(settings,map_seed,planner_seed):
    """Return the preregistered five-planner Latin row for a problem slot."""
    try:
        map_index=settings["map_seeds"].index(map_seed)
        seed_index=settings["planner_seeds"].index(planner_seed)
    except ValueError as error:
        raise ValueError("combined slot contains an unregistered map/planner seed") from error
    return (map_index+seed_index)%len(COMBINED_PLANNERS)


def _combined_order(settings,map_seed,planner_seed):
    return COMBINED_LATIN_SQUARE[_combined_row_index(settings,map_seed,planner_seed)]


def audit_combined_plan(trials,settings):
    """Verify complete five-planner blocks and the preregistered Latin balance.

    Block order may be shuffled, but a block's row is a pure function of the
    declared map-seed and planner-seed ranks.  It therefore cannot change in
    response to a planner outcome or to the shuffle order.
    """
    size=len(COMBINED_PLANNERS)
    if len(trials)%size:
        raise ValueError("combined plan ends inside a five-planner block")
    positions={planner:[0]*size for planner in COMBINED_PLANNERS}
    family_positions={family:{planner:[0]*size for planner in COMBINED_PLANNERS}
                      for family in settings["families"]}
    row_counts={str(row):0 for row in range(size)}
    family_row_counts={family:{str(row):0 for row in range(size)}
                       for family in settings["families"]}
    slots=[]
    for offset in range(0,len(trials),size):
        group=trials[offset:offset+size]
        keys={(entry["map_id"],entry["map_hash"],entry["query_id"],entry["query_hash"],
              entry["planner_seed"],tuple(entry["start"]),tuple(entry["goal"])) for entry in group}
        if len(keys)!=1 or len({entry.get("problem_slot_id") for entry in group})!=1:
            raise ValueError("combined problem slot identity mismatch")
        leader=group[0]
        if leader.get("family") not in family_positions:
            raise ValueError("combined problem slot contains an unregistered family")
        row=_combined_row_index(settings,leader["map_seed"],leader["planner_seed"])
        expected=COMBINED_LATIN_SQUARE[row]
        if tuple(entry["planner_id"] for entry in group)!=expected:
            raise ValueError("combined preregistered Latin row mismatch")
        if [entry.get("problem_slot_order") for entry in group]!=[offset//size]*size:
            raise ValueError("combined problem slot order mismatch")
        if [entry.get("within_slot_order") for entry in group]!=list(range(size)):
            raise ValueError("combined within-slot order mismatch")
        for index,entry in enumerate(group):
            positions[entry["planner_id"]][index]+=1
            family_positions[leader["family"]][entry["planner_id"]][index]+=1
        row_counts[str(row)]+=1
        family_row_counts[leader["family"]][str(row)]+=1
        slots.append(leader["problem_slot_id"])
    if len(slots)!=len(set(slots)):
        raise ValueError("duplicate combined problem slot")
    if [entry.get("shuffle_order") for entry in trials]!=list(range(len(trials))):
        raise ValueError("non-contiguous combined shuffle order")
    if len(slots)%size:
        raise ValueError("combined problem slots do not contain complete Latin cycles")
    global_target=len(slots)//size
    if any(count!=global_target for counts in positions.values() for count in counts):
        raise ValueError("combined global position counts are not exactly balanced")
    if any(count!=global_target for count in row_counts.values()):
        raise ValueError("combined global Latin-row counts are not exactly balanced")
    family_targets={family:sum(1 for offset in range(0,len(trials),size)
                               if trials[offset]["family"]==family)//size
                    for family in settings["families"]}
    if any(sum(1 for offset in range(0,len(trials),size)
               if trials[offset]["family"]==family)%size for family in settings["families"]):
        raise ValueError("combined family slots do not contain complete Latin cycles")
    for family,target in family_targets.items():
        if any(count!=target for counts in family_positions[family].values() for count in counts):
            raise ValueError("combined family position counts are not exactly balanced")
        if any(count!=target for count in family_row_counts[family].values()):
            raise ValueError("combined family Latin-row counts are not exactly balanced")
    return {"policy":TRIAL_ORDER_COMBINED,"planned_trials":len(trials),
            "problem_slots":len(slots),"trials_per_problem_slot":size,
            "latin_square":list(map(list,COMBINED_LATIN_SQUARE)),
            "position_counts":positions,"family_position_counts":family_positions,
            "latin_row_counts":row_counts,"family_latin_row_counts":family_row_counts,
            "row_assignment":"(map_seed_rank + planner_seed_rank) modulo 5 before block shuffle",
            "complete_planner_blocks":True}


def plan(config,profile,map_manifest,*,order_policy=TRIAL_ORDER_INDIVIDUAL,max_problem_slots=None):
    if config.get("schema_version") in (8,9):
        if config["schema_version"] == 9:
            from a3_traversability_perf_protocol import PROTOCOL_ID,validate_config
        else:
            from a3_traversability_protocol import PROTOCOL_ID,validate_config
        validate_config(config,profile)
        if (map_manifest.get("config_hash")!=digest(config) or
                map_manifest.get("profile")!="smoke" or map_manifest.get("protocol_id")!=PROTOCOL_ID):
            raise ValueError("traversability map manifest protocol/config mismatch")
    trials=[];settings=config["profiles"][profile]
    if order_policy==TRIAL_ORDER_PAIRED and set(settings["planners"])!=set(PAIRED_PLANNERS):
        raise ValueError("paired problem-slot ordering requires the four registered planners")
    if order_policy==TRIAL_ORDER_LATIN and tuple(settings["planners"])!=ADVANCED_PLANNERS:
        raise ValueError("Latin-square ordering requires the four locked advanced planners")
    if (order_policy==TRIAL_ORDER_SEM_COMPARISON and
            tuple(settings["planners"])!=SEM_COMPARISON_PLANNERS):
        raise ValueError("SEM comparison ordering requires the two locked planners")
    if (order_policy==TRIAL_ORDER_COMBINED and
            tuple(settings["planners"])!=COMBINED_PLANNERS):
        raise ValueError("combined ordering requires the five locked planners")
    expected_maps={(family,seed) for family in settings["families"] for seed in settings["map_seeds"]}
    if {(m["family"],m["map_seed"]) for m in map_manifest["maps"]}!=expected_maps or len(map_manifest["maps"])!=len(expected_maps):
        raise ValueError("map manifest does not match profile")
    for m in map_manifest["maps"]:
        if len(m["queries"])!=settings["queries_per_map"] or len({q["query_id"] for q in m["queries"]})!=len(m["queries"]):
            raise ValueError("query cardinality mismatch")
        for query in m["queries"]:
            if profile in ("pilot","advanced","sem_comparison","combined") and (query.get("feasibility")!="feasible_witness_validated" or not query.get("reference",{}).get("validation",{}).get("valid")):
                raise ValueError(f"{profile} feasible query lacks validated witness")
            if profile=="advanced" and query.get("query_id") not in config["advanced_benchmark"]["required_query_ids"]:
                raise ValueError("advanced manifest contains an unregistered query")
            if profile=="sem_comparison" and query.get("query_id") not in config["sem_comparison"]["required_query_ids"]:
                raise ValueError("SEM comparison manifest contains an unregistered query")
            if profile=="combined" and query.get("query_id") not in config["combined_benchmark"]["required_query_ids"]:
                raise ValueError("combined manifest contains an unregistered query")
            for planner in settings["planners"]:
                for seed in settings["planner_seeds"]:
                    trials.append({"trial_id":f'{m["map_id"]}_{query["query_id"]}_{planner}_s{seed}',
                        "map_id":m["map_id"],"map_path":m["path"],"map_hash":m["sha256"],"family":m["family"],
                        "map_seed":m["map_seed"],"query_id":query["query_id"],"start":query["start"],"goal":query["goal"],
                        "query_hash":query.get("query_hash",digest({"map_hash":m["sha256"],"start":query["start"],"goal":query["goal"],"model":config["model"]})),
                        "reference":query.get("reference"),"geometry_label":query.get("geometry_label","smoke_control"),
                        "planner_id":planner,"planner_seed":seed})
                    if config.get("planner_variants"):
                        options=resolved_planner_options(config,planner)
                        trials[-1].update(backend_planner_id=options["planner_id"],
                                          resolved_optimizations=options,
                                          resolved_optimizations_sha256=digest(options))
                    if config.get("schema_version") in (8,9):
                        from a3_traversability_protocol import objective_identity
                        identity=objective_identity(config,planner)
                        trials[-1].update(objective_type=identity["objective_type"],
                                          resolved_traversability=identity["traversability"],
                                          objective_config_sha256=digest(identity))
    if len({t["trial_id"] for t in trials})!=len(trials):raise ValueError("duplicate trial ids")
    expected=len(expected_maps)*settings["queries_per_map"]*len(settings["planners"])*len(settings["planner_seeds"])
    if len(trials)!=expected or (profile=="pilot" and expected!=400):raise ValueError("trial cardinality mismatch")
    if profile=="advanced" and expected!=config["advanced_benchmark"]["expected_trials"]:
        raise ValueError("advanced trial cardinality mismatch")
    if profile=="sem_comparison" and expected!=config["sem_comparison"]["expected_trials"]:
        raise ValueError("SEM comparison trial cardinality mismatch")
    if profile=="combined" and expected!=config["combined_benchmark"]["expected_trials"]:
        raise ValueError("combined trial cardinality mismatch")
    rng=random.Random(config["execution"]["shuffle_trial_order_seed"])
    if order_policy==TRIAL_ORDER_INDIVIDUAL:
        if max_problem_slots is not None:raise ValueError("max_problem_slots requires paired problem-slot ordering")
        rng.shuffle(trials)
        for index,t in enumerate(trials):t["shuffle_order"]=index
        return trials
    if order_policy==TRIAL_ORDER_LATIN:
        if max_problem_slots is not None:
            raise ValueError("locked advanced plan does not permit trial/seed substitution or a limited subset")
        grouped={}
        for trial in trials:grouped.setdefault((trial["map_id"],trial["query_id"],trial["planner_seed"]),[]).append(trial)
        slots=sorted(grouped.items(),key=lambda item:item[0])
        for _,group in slots:
            if {entry["planner_id"] for entry in group}!=set(ADVANCED_PLANNERS) or len({entry["query_hash"] for entry in group})!=1:
                raise ValueError("incomplete or inconsistent Latin-square problem slot")
        rng.shuffle(slots)
        ordered=[]
        for slot_order,(slot_key,group) in enumerate(slots):
            by_planner={entry["planner_id"]:entry for entry in group}
            problem_slot_id=f"{slot_key[0]}_{slot_key[1]}_s{slot_key[2]}"
            for within_slot_order,planner in enumerate(ADVANCED_LATIN_SQUARE[slot_order%len(ADVANCED_LATIN_SQUARE)]):
                entry=by_planner[planner]
                entry.update(problem_slot_id=problem_slot_id,problem_slot_order=slot_order,
                             within_slot_order=within_slot_order,trial_order_policy=order_policy)
                ordered.append(entry)
        for index,trial in enumerate(ordered):trial["shuffle_order"]=index
        audit_latin_plan(ordered)
        return ordered
    if order_policy==TRIAL_ORDER_SEM_COMPARISON:
        if max_problem_slots is not None:
            raise ValueError("locked SEM comparison plan does not permit a limited subset")
        grouped={}
        for trial in trials:
            grouped.setdefault((trial["map_id"],trial["query_id"],trial["planner_seed"]),[]).append(trial)
        slots=sorted(grouped.items(),key=lambda item:item[0])
        for _,group in slots:
            if ({entry["planner_id"] for entry in group}!=set(SEM_COMPARISON_PLANNERS) or
                    len({entry["query_hash"] for entry in group})!=1):
                raise ValueError("incomplete or inconsistent SEM comparison problem slot")
        rng.shuffle(slots);ordered=[]
        for slot_order,(slot_key,group) in enumerate(slots):
            by_planner={entry["planner_id"]:entry for entry in group}
            problem_slot_id=f"{slot_key[0]}_{slot_key[1]}_s{slot_key[2]}"
            for within_slot_order,planner in enumerate(
                    _sem_comparison_order(settings,group[0]["map_seed"],slot_key[2])):
                entry=by_planner[planner]
                entry.update(problem_slot_id=problem_slot_id,problem_slot_order=slot_order,
                             within_slot_order=within_slot_order,trial_order_policy=order_policy)
                ordered.append(entry)
        for index,trial in enumerate(ordered):trial["shuffle_order"]=index
        audit_sem_comparison_plan(ordered,settings)
        return ordered
    if order_policy==TRIAL_ORDER_COMBINED:
        if max_problem_slots is not None:
            raise ValueError("locked combined plan does not permit a limited subset")
        grouped={}
        for trial in trials:
            grouped.setdefault((trial["map_id"],trial["query_id"],trial["planner_seed"]),[]).append(trial)
        slots=sorted(grouped.items(),key=lambda item:item[0])
        for _,group in slots:
            if ({entry["planner_id"] for entry in group}!=set(COMBINED_PLANNERS) or
                    len({entry["query_hash"] for entry in group})!=1):
                raise ValueError("incomplete or inconsistent combined problem slot")
        # Only block order is shuffled.  The row itself is assigned from the
        # declared seed ranks and is independent of this randomized order.
        rng.shuffle(slots);ordered=[]
        for slot_order,(slot_key,group) in enumerate(slots):
            by_planner={entry["planner_id"]:entry for entry in group}
            problem_slot_id=f"{slot_key[0]}_{slot_key[1]}_s{slot_key[2]}"
            for within_slot_order,planner in enumerate(
                    _combined_order(settings,group[0]["map_seed"],slot_key[2])):
                entry=by_planner[planner]
                entry.update(problem_slot_id=problem_slot_id,problem_slot_order=slot_order,
                             within_slot_order=within_slot_order,trial_order_policy=order_policy)
                ordered.append(entry)
        for index,trial in enumerate(ordered):trial["shuffle_order"]=index
        audit_combined_plan(ordered,settings)
        return ordered
    if order_policy!=TRIAL_ORDER_PAIRED:raise ValueError("unknown trial order policy: "+str(order_policy))
    if set(settings["planners"])!=set(PAIRED_PLANNERS):
        raise ValueError("paired problem-slot ordering requires the four registered planners")
    grouped={}
    for trial in trials:grouped.setdefault((trial["map_id"],trial["query_id"],trial["planner_seed"]),[]).append(trial)
    slots=sorted(grouped.items(),key=lambda item:item[0])
    for _,group in slots:
        if {entry["planner_id"] for entry in group}!=set(PAIRED_PLANNERS) or len({entry["query_hash"] for entry in group})!=1:
            raise ValueError("incomplete or inconsistent paired problem slot")
    if max_problem_slots is None:rng.shuffle(slots)
    else:slots=_balanced_problem_slot_subset(slots,settings,max_problem_slots,rng)
    ordered=[]
    for slot_order,(slot_key,group) in enumerate(slots):
        by_planner={entry["planner_id"]:entry for entry in group}
        problem_slot_id=f"{slot_key[0]}_{slot_key[1]}_s{slot_key[2]}"
        for within_slot_order,planner in enumerate(PAIRED_PLANNER_CYCLE[slot_order%len(PAIRED_PLANNER_CYCLE)]):
            entry=by_planner[planner]
            entry.update(problem_slot_id=problem_slot_id,problem_slot_order=slot_order,
                         within_slot_order=within_slot_order,trial_order_policy=order_policy)
            ordered.append(entry)
    for index,trial in enumerate(ordered):trial["shuffle_order"]=index
    audit_paired_plan(ordered)
    return ordered

def request_for(entry,config,profile):
    settings=config["profiles"][profile];algorithm=config["algorithm"]
    request={"map_path":entry["map_path"],"planner_id":entry["planner_id"],"planner_seed":entry["planner_seed"],
        "start":entry["start"],"goal":entry["goal"],"planning_budget_s":settings["planning_budget_s"],
        "samples_per_batch":algorithm["bitstar_samples_per_batch"],"rewire_factor":algorithm["bitstar_rewire_factor"],
        "sample_attempts":algorithm["informed_sample_attempts_per_call"],"progress_interval_s":config["logging"]["progress_interval_s"]}
    if entry["planner_id"] in config.get("planner_variants",{}):
        options=resolved_planner_options(config,entry["planner_id"])
        if (entry.get("resolved_optimizations",options)!=options or
                entry.get("resolved_optimizations_sha256",digest(options))!=digest(options)):
            raise ValueError("resolved planner options changed after trial planning")
        request.update(options,variant_id=entry["planner_id"])
    if config.get("schema_version") in (8,9):
        from a3_traversability_protocol import objective_identity
        identity=objective_identity(config,entry["planner_id"])
        if (entry.get("objective_config_sha256",digest(identity))!=digest(identity) or
                entry.get("resolved_traversability",identity["traversability"])!=identity["traversability"]):
            raise ValueError("objective configuration changed after trial planning")
        request.update(identity)
    if entry["planner_id"]=="sem_rrtstar_common_v1":
        protocol=(config["combined_benchmark"] if profile=="combined"
                  else config["sem_comparison"])
        request.update(sem_range_m=protocol["sem_range_m"],
                       sem_neighbor_radius_m=protocol["sem_neighbor_radius_m"])
    for key in ("work_iterations","audit_tree"):
        if key in entry:request[key]=entry[key]
    return request


def verify_variant_result(entry,config,record):
    """Reject a worker that silently ignores a requested variant or relabels A3."""
    if config.get("schema_version") in (8,9):
        from a3_traversability_protocol import objective_identity
        identity=objective_identity(config,entry["planner_id"])
        if (record.get("objective_type")!=identity["objective_type"] or
                record.get("requested_traversability")!=identity["traversability"] or
                record.get("effective_traversability")!=identity["traversability"]):
            raise ValueError("worker did not apply requested traversability objective")
    if entry["planner_id"] not in config.get("planner_variants",{}):return
    options=resolved_planner_options(config,entry["planner_id"])
    if (record.get("planner_id")!=entry["planner_id"] or
            record.get("variant_id")!=entry["planner_id"] or
            record.get("backend_planner_id")!=options["planner_id"]):
        raise ValueError("worker variant/backend identity differs from request")
    # An invalid query returns before setup and cannot establish planner efficacy.
    if (record.get("termination_reason")=="INVALID_QUERY" and
            record.get("planner_status") in {"INVALID_START","INVALID_GOAL"}):return
    effective=record.get("effective_policy",{})
    if config.get("schema_version") == 9:
        for key in ("sampler_query_cache", "traversability_build_policy", "exact_flat_certificate"):
            if effective.get(key) != options[key] or type(effective.get(key)) is not type(options[key]):
                raise ValueError(f"worker did not apply requested {key}")
        certificate=record.get("exact_certificate",{})
        if (certificate.get("policy") != "flat_direct_v1" or
                certificate.get("enabled") is not options["exact_flat_certificate"]):
            raise ValueError("worker exact certificate policy/provenance missing")
    for key in ("state_cache","scalar_cost","collision_index"):
        if effective.get(key) is not options[key]:
            raise ValueError(f"worker did not apply requested {key}")
    if options["planner_id"] in {"bitstar_a0","bitstar_a3","abitstar","aitstar"}:
        if effective.get("samples_per_batch")!=options["samples_per_batch"]:
            raise ValueError("worker initial batch size differs from request")
    if effective.get("batch_after_first_solution")!=options["batch_after_first_solution"]:
        raise ValueError("worker batch schedule differs from request")
    expected_policy="after_first_solution" if options["batch_after_first_solution"] else "fixed"
    if effective.get("batch_policy")!=expected_policy:
        raise ValueError("worker batch policy differs from request")


def run_trial(entry,config,directory,worker,env,experiment_id,config_hash,*,profile="smoke"):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    request=request_for(entry,config,profile)
    dump(directory/"request.json",request);dump(directory/"resolved_config.json",config)
    trial=dict(entry,schema_version=2,experiment_id=experiment_id,config_hash=config_hash,environment=env,
        code_sha256=env["code_sha256"],code_commit=env["code_commit"],utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        execution_status="BLOCKED",planner_status="NOT_RUN",independent_valid=None,success=False,path_available=False,
        time_budget_s=request["planning_budget_s"],best_cost_terrain_m=None,first_solution_time_s=None)
    command=[str(worker),str((directory/"request.json").resolve()),str(directory.resolve())]
    trial["command"]=command;trial["attempt_history"]=[{"utc":utc(),"event":"START_REQUESTED","command":command}]
    dump(directory/"attempt.json",{"entry":entry,"command":command,"utc":utc(),"status":"START_REQUESTED"})
    began=time.perf_counter();peak=0;exit_code=None;failure=None;proc=None;interrupted=False
    with (directory/"stdout.log").open("wb") as stdout,(directory/"stderr.log").open("wb") as stderr:
        try:
            proc=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=worker_env())
            trial["attempt_history"].append({"utc":utc(),"event":"PROCESS_STARTED","pid":proc.pid})
            ps=psutil.Process(proc.pid)
            while proc.poll() is None:
                try:peak=max(peak,ps.memory_info().rss/2**20)
                except psutil.NoSuchProcess:pass
                if peak>config["execution"]["memory_limit_mb"]:
                    failure="MEMORY_LIMIT";proc.kill();break
                if time.perf_counter()-began>request["planning_budget_s"]+config["execution"]["watchdog_grace_s"]:
                    failure="WATCHDOG";proc.kill();break
                time.sleep(.02)
            exit_code=proc.wait()
        except OSError as error:
            failure="BLOCKED";stderr.write(str(error).encode())
        except KeyboardInterrupt:
            failure="INTERRUPTED";interrupted=True
            if proc is not None and proc.poll() is None:proc.kill()
            if proc is not None:exit_code=proc.wait()
    if (directory/"worker.json").exists():
        try:
            record=read_json(directory/"worker.json")
            try:verify_variant_result(entry,config,record)
            except ValueError as error:
                failure="INEFFECTIVE_REQUESTED_VARIANT";trial["worker_policy_error"]=str(error)
            else:trial.update(record)
        except (ValueError,OSError) as error:failure="MALFORMED_WORKER_JSON";trial["worker_read_error"]=str(error)
    if failure:
        trial["execution_status"]={"BLOCKED":"BLOCKED","WATCHDOG":"WATCHDOG","MEMORY_LIMIT":"WATCHDOG","INTERRUPTED":"CRASH"}.get(failure,"CRASH")
        trial["termination_reason"]=failure
    elif exit_code!=0:trial["execution_status"]="CRASH";trial["termination_reason"]="NONZERO_WORKER_EXIT"
    elif not (directory/"worker.json").exists():trial["execution_status"]="CRASH";trial["termination_reason"]="MISSING_WORKER_RESULT"
    trial["exit_code"]=exit_code;trial["peak_rss_mb"]=peak if exit_code is not None else None
    trial["worker_wall_s"]=time.perf_counter()-began
    trial["rss_measurement"]="sampled_peak_20ms_not_OS_exact_peak"
    trial["attempt_history"].append({"utc":utc(),"event":"INTERRUPTED" if interrupted else "PROCESS_FINISHED","exit_code":exit_code,"reason":failure})
    if (directory/"ompl.log").exists() and request["planner_id"].startswith("bitstar"):
        text=(directory/"ompl.log").read_text(errors="replace")
        matches=re.findall(r"Pruning disconnected (\d+) vertices.*?removed (\d+) samples",text)
        trial["graph_prunings_log_observed"]=len(matches)
        trial["vertices_disconnected_log_observed"]=sum(int(a) for a,b in matches)
        trial["samples_pruned_log_observed"]=sum(int(b) for a,b in matches)
        trial.update(bitstar_log_counters(text))
    dump(directory/"trial.json",trial)
    if interrupted:raise KeyboardInterrupt
    return trial

def summarize(output,manifest):
    trials=[read_json(output/t["trial_id"]/"trial.json") for t in manifest["trials"] if (output/t["trial_id"]/"trial.json").exists()]
    executed=sum(t.get("execution_status")!="BLOCKED" for t in trials)
    invalid=sum(bool(t.get("path_available")) and (t.get("independent_valid") is False or t.get("snapshots_valid") is False) for t in trials)
    summary={"planned":len(manifest["trials"]),"executed":executed,"validated_success":sum(bool(t.get("success")) for t in trials),
        "timeout":sum(t.get("execution_status")=="TIMEOUT" for t in trials),"invalid":invalid,
        "crash":sum(t.get("execution_status")=="CRASH" for t in trials),"watchdog":sum(t.get("execution_status")=="WATCHDOG" for t in trials),
        "blocked":sum(t.get("execution_status")=="BLOCKED" for t in trials),"not_run":len(manifest["trials"])-executed,
        "interrupted":sum(t.get("termination_reason","").startswith("INTERRUPTED") for t in trials)}
    summary["success"]=summary["validated_success"];summary["unexecuted"]=summary["not_run"]
    return summary

def refresh_manifest(output,manifest,*,validated=False):
    manifest["counts"]=summarize(output,manifest);manifest["updated_utc"]=utc()
    if manifest["counts"]["not_run"]:manifest["status"]="PARTIAL" if manifest["counts"]["executed"] or manifest["counts"]["blocked"] else "PLANNED"
    else:manifest["status"]="COMPLETE_VALIDATED" if validated else "EXECUTED_PENDING_VALIDATION"
    if validated and manifest["counts"]["invalid"]:manifest["status"]="FAILED_VALIDATION"
    dump(output/"manifest.json",manifest)

def provenance_identity(config,maps,worker):
    dependencies=read_json(ROOT/"configs/dependencies.lock.json");archives={}
    for name,record in dependencies["archives"].items():
        actual=file_hash(ROOT/".deps"/(name+".tar.gz"))
        if actual!=record["sha256"]:raise ValueError("dependency archive changed: "+name)
        archives[name]=actual
    return {"config_hash":digest(config),"code_sha256":code_hash(),"worker_sha256":file_hash(worker) if worker.exists() else None,
        "map_hashes":{m["map_id"]:m["sha256"] for m in maps["maps"]},"map_manifest_sha256":digest(maps),
        "dependency_lock_sha256":file_hash(ROOT/"configs/dependencies.lock.json"),"dependency_archives":archives,
        "patch_sha256":{str(p.relative_to(ROOT)):file_hash(p) for p in sorted((ROOT/"patches").glob("*.patch"))},
        "ompl_library_sha256":file_hash(ROOT/".deps/install/lib/libompl.dll") if (ROOT/".deps/install/lib/libompl.dll").exists() else None}

def archive_source(output,identity,config_path=None):
    paths=source_files()+[Path(config_path or ROOT/"configs/experiments.yaml"),ROOT/"configs/dependencies.lock.json"]
    paths=list(dict.fromkeys(Path(path).resolve() for path in paths))
    records={str(p.relative_to(ROOT)):file_hash(p) for p in paths}
    with zipfile.ZipFile(output/"source_at_run.zip","x",zipfile.ZIP_DEFLATED) as archive:
        for path in paths:archive.write(path,path.relative_to(ROOT).as_posix())
    dump(output/"source_manifest.json",{"code_sha256":identity["code_sha256"],"files":records,
        "archive_sha256":file_hash(output/"source_at_run.zip"),"hash_policy":"strict entire source set including analysis/tests/diagnostics; no relaxation on resume"})

def verify_resume_identity(previous,current):
    if previous!=current:raise ValueError("resume identity mismatch: config/map/code/dependency/patch/binary changed")

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--profile",default="smoke")
    p.add_argument("--output",required=True);p.add_argument("--jobs",type=int,default=1);p.add_argument("--worker",type=Path,default=default_worker())
    p.add_argument("--map-manifest",type=Path);p.add_argument("--dry-run",action="store_true");p.add_argument("--resume",action="store_true")
    p.add_argument("--order-policy",choices=(TRIAL_ORDER_INDIVIDUAL,TRIAL_ORDER_PAIRED,
                                              TRIAL_ORDER_LATIN,TRIAL_ORDER_SEM_COMPARISON,
                                              TRIAL_ORDER_COMBINED),
                   default=TRIAL_ORDER_INDIVIDUAL)
    p.add_argument("--max-trials",type=int,help="paired limited run only; must contain complete problem slots")
    p.add_argument("--allow-full",action="store_true",help="explicitly allow a guarded limited run using the full-profile inputs")
    p.add_argument("--allow-advanced-benchmark",action="store_true",help="explicitly allow the locked A3-vs-advanced dry-run plan")
    p.add_argument("--allow-sem-comparison",action="store_true",help="explicitly allow the locked A3-vs-SEM common-objective dry-run plan")
    p.add_argument("--allow-combined-benchmark",action="store_true",help="explicitly allow the locked five-planner combined dry-run plan")
    args=p.parse_args()
    if args.jobs!=1:raise ValueError("G3-G4 requires jobs=1")
    if args.profile=="full":
        if not args.allow_full:raise ValueError("full-profile inputs require explicit --allow-full")
        if args.order_policy!=TRIAL_ORDER_PAIRED or args.max_trials is None:
            raise ValueError("generic full-profile execution is limited to an explicit paired subset")
        config=load_config(args.config,"pilot")
    else:config=load_config(args.config,args.profile)
    if args.profile=="advanced":
        if not args.allow_advanced_benchmark:
            raise ValueError("advanced benchmark requires explicit --allow-advanced-benchmark")
        if args.order_policy!=TRIAL_ORDER_LATIN or args.max_trials is not None:
            raise ValueError("advanced benchmark requires the complete locked Latin-square plan")
        if not args.dry_run:
            raise ValueError("execute the timed campaign with tools/run_advanced_campaign.py")
    if args.profile=="sem_comparison":
        if not args.allow_sem_comparison:
            raise ValueError("SEM comparison requires explicit --allow-sem-comparison")
        if args.order_policy!=TRIAL_ORDER_SEM_COMPARISON or args.max_trials is not None:
            raise ValueError("SEM comparison requires the complete locked paired parity plan")
        if not args.dry_run:
            raise ValueError("execute the timed campaign with tools/run_sem_comparison_campaign.py")
    if args.profile=="combined":
        if not args.allow_combined_benchmark:
            raise ValueError("combined benchmark requires explicit --allow-combined-benchmark")
        if args.order_policy!=TRIAL_ORDER_COMBINED or args.max_trials is not None:
            raise ValueError("combined benchmark requires the complete locked five-planner Latin plan")
        if not args.dry_run:
            raise ValueError("execute the timed campaign with tools/run_combined_campaign.py")
    planner_count=len(config["profiles"][args.profile]["planners"])
    if args.max_trials is not None and (args.max_trials<=0 or args.max_trials%planner_count):
        raise ValueError("max-trials must be positive and contain complete problem slots")
    max_problem_slots=args.max_trials//planner_count if args.max_trials is not None else None
    config_hash=digest(config);output=Path(args.output)
    protected=(ROOT/"results/smoke").resolve()
    if output.resolve()==protected or protected in output.resolve().parents:raise ValueError("completed smoke is immutable; choose a new experiment_id")
    maps=read_json(args.map_manifest or ROOT/"data"/args.profile/"manifest.json")
    if maps["config_hash"]!=config_hash:raise ValueError("generate maps for this exact config first")
    for m in maps["maps"]:
        if file_hash(m["path"])!=m["sha256"]:raise ValueError("map data changed")
    trials=plan(config,args.profile,maps,order_policy=args.order_policy,max_problem_slots=max_problem_slots)
    if args.max_trials is not None and len(trials)!=args.max_trials:raise ValueError("planned trial count differs from explicit limit")
    env=environment();experiment_id=output.name
    identity=provenance_identity(config,maps,args.worker)
    schedule=(audit_paired_plan(trials,160) if args.order_policy==TRIAL_ORDER_PAIRED else
              audit_latin_plan(trials) if args.order_policy==TRIAL_ORDER_LATIN else
              audit_sem_comparison_plan(trials,config["profiles"][args.profile])
              if args.order_policy==TRIAL_ORDER_SEM_COMPARISON else
              audit_combined_plan(trials,config["profiles"][args.profile])
              if args.order_policy==TRIAL_ORDER_COMBINED else
              {"policy":TRIAL_ORDER_INDIVIDUAL,"planned_trials":len(trials)})
    schedule.update(shuffle_seed=config["execution"]["shuffle_trial_order_seed"],requested_max_trials=args.max_trials,
                    classification="LIMITED_PAIRED_RUN_NOT_FULL_G5" if args.profile=="full" else
                     "LOCKED_A3_VS_ADVANCED_V1" if args.profile=="advanced" else
                     "LOCKED_A3_VS_SEM_RRTSTAR_COMMON_V1" if args.profile=="sem_comparison" else
                     "LOCKED_A3_VS_ALL_V1" if args.profile=="combined" else
                     "STANDARD_PROFILE")
    identity.update(trial_order_policy=args.order_policy,schedule_sha256=digest(schedule))
    manifest={"schema_version":2,"experiment_id":experiment_id,"profile":args.profile,"identity":identity,
        "planned_trials":len(trials),"trials":trials,"status":"PLANNED","history":[],"created_utc":utc(),
        "schedule":schedule,
        "resume_policy":"strict config/map/model/entire-source/dependency/patch/binary hashes; interrupted attempts retained without rerun",
        "config_field_scope":{"roughness_window_m":"inactive_because_roughness_limit_m_is_null",
            "logging.save_ompl_benchmark_log":"raw_OMPL_OutputHandler_log_not_Planner_Arena"}}
    if (output/"manifest.json").exists():
        old=read_json(output/"manifest.json")
        verify_resume_identity(old["identity"],identity)
        if file_hash(output/"source_at_run.zip")!=read_json(output/"source_manifest.json")["archive_sha256"]:
            raise ValueError("source archive changed")
        if not args.resume and old["status"]!="PLANNED":raise ValueError("output exists: use --resume or a new experiment directory")
        manifest=old
    else:
        if output.exists() and any(output.iterdir()):raise ValueError("nonempty output without manifest")
        output.mkdir(parents=True,exist_ok=True);archive_source(output,identity,args.config)
        dump(output/"environment.json",env);dump(output/"resolved_config.json",config);dump(output/"map_manifest.json",maps)
    manifest.setdefault("history",[]).append({"utc":utc(),"event":"DRY_RUN" if args.dry_run else ("RESUME" if args.resume else "RUN"),"identity":identity})
    refresh_manifest(output,manifest)
    print(f"planned={len(trials)}; profile={args.profile}; output={output}",flush=True)
    if args.dry_run:return 0
    # A correctness gate is generated by tests/test_pipeline.py only after its full
    # success. Tie it to the worker bytes; smoke cannot silently bypass that gate.
    gate_path=args.worker.resolve().parent/"correctness_gate.json"
    if not gate_path.exists() or read_json(gate_path).get("worker_sha256")!=identity["worker_sha256"] or read_json(gate_path).get("code_sha256")!=identity["code_sha256"]:
        raise RuntimeError("correctness gate absent/stale: run ctest before trials")
    try:
        for entry in trials:
            directory=output/entry["trial_id"]
            if (directory/"trial.json").exists():continue
            if directory.exists():
                history=read_json(directory/"attempt.json") if (directory/"attempt.json").exists() else None
                dump(directory/"trial.json",dict(entry,execution_status="CRASH",planner_status="UNKNOWN",success=False,path_available=False,
                    independent_valid=None,time_budget_s=config["profiles"][args.profile]["planning_budget_s"],termination_reason="INTERRUPTED_PREVIOUS_RUN",
                    experiment_id=experiment_id,config_hash=config_hash,attempt_history=[history,{"utc":utc(),"event":"INTERRUPTED_PREVIOUS_RUN_RETAINED"}]))
                refresh_manifest(output,manifest);continue
            result=run_trial(entry,config,directory,args.worker,env,experiment_id,config_hash,profile=args.profile)
            refresh_manifest(output,manifest)
            print(f'{entry["shuffle_order"]+1}/{len(trials)} {entry["trial_id"]}: {result["execution_status"]} / {result["planner_status"]}',flush=True)
    except KeyboardInterrupt:
        manifest["history"].append({"utc":utc(),"event":"RUN_INTERRUPTED"});refresh_manifest(output,manifest)
        print("PARTIAL: interrupted; all existing attempt artifacts retained",flush=True);return 130
    refresh_manifest(output,manifest)
    return 0

if __name__=="__main__":raise SystemExit(main())
