#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Aggregate AUC and Mugration metrics from Slurm array output.")
    parser.add_argument("--outdir", default="replicate_results", help="Base output directory where replicate_* folders live.")
    args = parser.parse_args()

    outdir_base = Path(args.outdir)
    
    print(f"\n{'='*60}")
    print(" Aggregating AUC rankings across all replicates...")
    print(f"{'='*60}")
    
    # 1. AGGREGATE AUC RANKINGS
    auc_files = list(outdir_base.glob("replicate_*/AUC_rankings.csv"))
    if auc_files:
        df_list = []
        for f in auc_files:
            df = pd.read_csv(f)
            df["replicate"] = f.parent.name
            df_list.append(df)
            
        all_auc = pd.concat(df_list, ignore_index=True)
        agg_auc = all_auc.groupby(["eval_type", "algorithm", "scenario_id", "scenario_label"]).agg(
            median_auc=("auc", "median"),
            mean_auc=("auc", "mean"),
            median_rank=("rank_overall", "median"),
            successful_runs=("auc", "count")
        ).reset_index()
        
        agg_auc["rank_of_median_auc"] = agg_auc.groupby("eval_type")["median_auc"].rank(method="dense", ascending=True)
        agg_auc = agg_auc.sort_values(["eval_type", "rank_of_median_auc", "algorithm", "scenario_id"])
        
        auc_out = outdir_base / "Aggregated_Median_AUC_Rankings.csv"
        agg_auc.to_csv(auc_out, index=False)
        print(f"Saved: {auc_out}")
    else:
        print("No AUC_rankings.csv files found.")

    # 2. AGGREGATE MUGRATION METRICS
    print(f"\n{'='*60}")
    print(" Aggregating Mugration Metrics across all replicates...")
    print(f"{'='*60}")

    mug_files = list(outdir_base.glob("replicate_*/Mugration_Metrics.csv"))
    if mug_files:
        mug_list = []
        for f in mug_files:
            df = pd.read_csv(f)
            df["replicate"] = f.parent.name
            mug_list.append(df)
            
        all_mug = pd.concat(mug_list, ignore_index=True)
        agg_mug = all_mug.groupby(["algorithm", "scenario_id", "scenario_label"]).agg(
            median_pearson=("pearson_r", "median"),
            mean_pearson=("pearson_r", "mean"),
            median_mae=("mae", "median"),
            mean_mae=("mae", "mean"),
            successful_runs=("pearson_r", "count")
        ).reset_index()
        
        # Sort by Pearson (Highest is best)
        agg_mug = agg_mug.sort_values(["scenario_id", "median_pearson"], ascending=[True, False])
        
        mug_out = outdir_base / "Aggregated_Mugration_Metrics.csv"
        agg_mug.to_csv(mug_out, index=False)
        print(f"Saved: {mug_out}")
        
        print("\n=== Top Strategy per Scenario (by Median Pearson Correlation) ===")
        top_mug = agg_mug.loc[agg_mug.groupby('scenario_id')['median_pearson'].idxmax()]
        for _, row in top_mug.iterrows():
            print(f"  Scenario {row['scenario_id']} ({row['scenario_label']}):")
            print(f"    -> {row['algorithm']} - Median Pearson: {row['median_pearson']:.4f} (MAE: {row['median_mae']:.4f})")
    else:
        print("No Mugration_Metrics.csv files found. Was --abm_mugration used?")

if __name__ == "__main__":
    main()
