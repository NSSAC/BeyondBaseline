#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Aggregate AUC and Mugration metrics from Slurm array output.")
    # NEW: Explicit input directory parameter
    parser.add_argument("--input-dir", default="replicate_results", 
                        help="Path to the parent directory containing all the replicate_* folders.")
    parser.add_argument("--outdir", default=None, 
                        help="Where to save the aggregated CSVs. Defaults to the same folder as --input-dir.")
    args = parser.parse_args()

    input_base = Path(args.input_dir)
    outdir_base = Path(args.outdir) if args.outdir else input_base
    
    # Ensure output directory exists
    outdir_base.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f" Aggregating AUC rankings from: {input_base}")
    print(f"{'='*60}")
    
    # 1. AGGREGATE AUC RANKINGS
    auc_files = list(input_base.glob("replicate_*/AUC_rankings.csv"))
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
    print(f" Aggregating Mugration Metrics from: {input_base}")
    print(f"{'='*60}")

    mug_files = list(input_base.glob("replicate_*/Mugration_Metrics.csv"))
    if mug_files:
        mug_list = []
        for f in mug_files:
            df = pd.read_csv(f)
            df["replicate"] = f.parent.name
            mug_list.append(df)
            
        all_mug = pd.concat(mug_list, ignore_index=True)
        
        # CRITICAL FIX: Use 'masked_mae', 'cosine_similarity', and 'topological_f1'
        agg_mug = all_mug.groupby(["algorithm", "scenario_id", "scenario_label"]).agg(
            median_pearson=("pearson_r", "median"),
            median_cosine=("cosine_similarity", "median"),
            median_f1=("topological_f1", "median"),
            median_masked_mae=("masked_mae", "median"),
            successful_runs=("pearson_r", "count")
        ).reset_index()
        
        # Sort by F1-Score (as it's often the most robust indicator of detecting the right transmission bridges)
        agg_mug = agg_mug.sort_values(["scenario_id", "median_f1"], ascending=[True, False])
        
        mug_out = outdir_base / "Aggregated_Mugration_Metrics.csv"
        agg_mug.to_csv(mug_out, index=False)
        print(f"Saved: {mug_out}")
        
        print("\n=== Top Strategy per Scenario (by Median F1-Score) ===")
        top_mug = agg_mug.loc[agg_mug.groupby('scenario_id')['median_f1'].idxmax()]
        for _, row in top_mug.iterrows():
            print(f"  Scenario {row['scenario_id']} ({row['scenario_label']}):")
            print(f"    -> {row['algorithm']} - F1: {row['median_f1']:.4f} (Cosine: {row['median_cosine']:.4f}, MAE: {row['median_masked_mae']:.4f})")
    else:
        print("No Mugration_Metrics.csv files found. Was --abm_mugration used?")

if __name__ == "__main__":
    main()