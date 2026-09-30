#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Run scenarios across multiple replicates and aggregate AUC rankings.")
    parser.add_argument("--replicates-dir", required=True, help="Path to the directory containing replicate folders (e.g., data/replicate)")
    parser.add_argument("--population", required=True, help="Path to the population file (required by run_all_scenarios.py)")
    parser.add_argument("--outdir", default="replicate_results", help="Base output directory for all runs (default: replicate_results)")
    parser.add_argument("--no-plots", action="store_true", help="Disable plotting for individual runs to save time and disk space")
    
    args = parser.parse_args()

    replicates_dir = Path(args.replicates_dir)
    outdir_base = Path(args.outdir)
    outdir_base.mkdir(parents=True, exist_ok=True)

    # 1. Find all replicate folders
    replicate_folders = [d for d in replicates_dir.iterdir() if d.is_dir() and d.name.startswith("replicate_")]
    
    if not replicate_folders:
        print(f"No replicate folders found matching pattern 'replicate_*' in {replicates_dir}")
        return

    print(f"Found {len(replicate_folders)} replicate folders. Starting batch run...")

    # 2. Run the script for each replicate
    for rep_dir in sorted(replicate_folders):
        print(f"\n{'='*60}")
        print(f" Processing: {rep_dir.name}")
        print(f"{'='*60}")
        
        linelist_path = rep_dir / "linelist.csv.xz"
        infections_path = rep_dir / "linelist_allevents.csv.xz"
        
        # Verify files exist
        if not linelist_path.exists() or not infections_path.exists():
            print(f"  [Warning] Missing linelist or infections file in {rep_dir}. Skipping.")
            continue
            
        rep_outdir = outdir_base / rep_dir.name
        
        # Build the command using the same Python interpreter currently running
        cmd = [
            sys.executable, "-m", "scenarios_simulation.run_all_scenarios",
            "--linelist", str(linelist_path),
            "--population", args.population,
            "--infections", str(infections_path),
            "--outdir", str(rep_outdir),
            "--batch-size", "100",
            "--no-replacement",
            "--seed", "42",
            "--algorithms", "surs", "stratified", "LASSO-Greedy", "LASSO-Stratified",
            "--stratifiers", "age", "race", "county", "sex"
        ]
        
        # Keep the no-plots flag logic if you still want to skip image generation 
        # during the massive batch run to save time
        if args.no_plots:
            cmd.append("--no-plots")
            
        try:
            # Execute the script and wait for it to finish
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"  [Error] Failed while running {rep_dir.name}. Skipping to next.")
            continue

    # 3. Aggregate AUC rankings
    print(f"\n{'='*60}")
    print(" Aggregating AUC rankings across all replicates...")
    print(f"{'='*60}")
    
    auc_files = list(outdir_base.glob("replicate_*/AUC_rankings.csv"))
    
    if not auc_files:
        print("No AUC_rankings.csv files found to aggregate. Did the runs fail?")
        return
        
    df_list = []
    for f in auc_files:
        df = pd.read_csv(f)
        df["replicate"] = f.parent.name
        df_list.append(df)
        
    all_auc = pd.concat(df_list, ignore_index=True)
    
    # Calculate median AUC and median Rank per evaluation type, algorithm, and scenario
    agg_df = all_auc.groupby(["eval_type", "algorithm", "scenario_id", "scenario_label"]).agg(
        median_auc=("auc", "median"),
        mean_auc=("auc", "mean"),
        median_rank=("rank_overall", "median"),
        successful_runs=("auc", "count")
    ).reset_index()
    
    # Re-calculate a new overall rank based on the consolidated median AUCs
    agg_df["rank_of_median_auc"] = agg_df.groupby("eval_type")["median_auc"].rank(method="dense", ascending=True)
    
    # Sort for readability
    agg_df = agg_df.sort_values(["eval_type", "rank_of_median_auc", "algorithm", "scenario_id"])
    
    final_out = outdir_base / "Aggregated_Median_AUC_Rankings.csv"
    agg_df.to_csv(final_out, index=False)
    
    print(f"\nDone! Aggregated results saved to: {final_out}")
    
    # Print a quick summary of the top strategy for each evaluation type
    print("\n=== Top Strategy per Evaluation Metric (by Median AUC) ===")
    top_strats = agg_df[agg_df["rank_of_median_auc"] == 1]
    for _, row in top_strats.iterrows():
        print(f"  {row['eval_type']}:")
        print(f"    -> {row['algorithm']} (Scenario {row['scenario_id']}) - Median AUC: {row['median_auc']:.4f}")

if __name__ == "__main__":
    main()