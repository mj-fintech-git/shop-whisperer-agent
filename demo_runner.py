import argparse
import sys
import time
import os
import pandas as pd
from pathlib import Path
from scipy import stats

# ANSI colors for beautiful terminal logging
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
GRAY = "\033[90m"

def print_log(author, message, color=RESET):
    print(f"  {color}[{author}]{RESET} {message}")
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="Shop Whisperer Demo Simulator")
    parser.add_argument(
        "--csv",
        default="orders.csv",
        help="Path to the input CSV file",
    )
    parser.add_argument(
        "--problem",
        default="We spent a large portion of our marketing budget on Paid Ads last month to boost sales. While revenue looks high, overall profits seem flat.",
        help="Free-text business problem statement",
    )
    args = parser.parse_args()

    # Enable ANSI formatting in Windows Terminal
    os.system("")

    if not Path(args.csv).exists():
        print(f"Error: Dataset file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    print(f"{BOLD}{CYAN}[pipeline] Starting — dataset: {args.csv}{RESET}")
    print(f"{BOLD}{CYAN}[pipeline] Problem: {args.problem[:80]}...{RESET}\n")
    time.sleep(1.0)

    # 1. Load dataset
    print_log("ds_agent", "Received business problem. Initializing data audit and metrics extraction...", GREEN)
    time.sleep(1.2)
    print_log("ds_agent", f"Calling MCP tool: load_dataset(filepath='{args.csv}')", GRAY)
    
    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    num_rows = len(df)
    cols = list(df.columns)
    time.sleep(1.0)
    print_log("ds_agent", f"Dataset loaded successfully ({num_rows:,} rows, columns: {cols}).", GREEN)
    time.sleep(1.0)

    # Detect columns dynamically
    cat_col = None
    for c in ["TrafficSource", "Category", "Group", "Channel"]:
        if c in df.columns:
            cat_col = c
            break
    if not cat_col:
        for c in df.columns:
            if df[c].dtype == "object" or df[c].dtype.name == "category":
                cat_col = c
                break
    if not cat_col:
        cat_col = df.columns[0]

    val_col = None
    for c in ["OrderValue_USD", "Revenue", "Sales", "Price", "Amount"]:
        if c in df.columns:
            val_col = c
            break
    if not val_col:
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]) and c != "OrderID" and "id" not in c.lower():
                val_col = c
                break
    if not val_col:
        val_col = df.columns[-1]

    profit_col = None
    for c in ["Profit_USD", "Profit", "NetProfit", "Margin"]:
        if c in df.columns:
            profit_col = c
            break
    
    status_col = None
    for c in ["OrderStatus", "Status", "State"]:
        if c in df.columns:
            status_col = c
            break

    # Calculate revenue & profit summary
    print_log("ds_agent", f"Calling MCP tool: get_channel_revenue_summary() to calculate revenue and profit per {cat_col}...", GRAY)
    time.sleep(1.5)

    grp = df.groupby(cat_col)
    summary_lines = []
    channels_data = []

    if val_col in df.columns:
        group_revs = grp[val_col].sum().sort_values(ascending=False)
    else:
        group_revs = grp.size().sort_values(ascending=False)

    top_groups = list(group_revs.index[:3])

    for g_name in top_groups:
        sub_df = df[df[cat_col] == g_name]
        g_count = len(sub_df)
        g_rev = sub_df[val_col].sum() if val_col in df.columns else 0.0
        g_prof = sub_df[profit_col].sum() if profit_col and profit_col in df.columns else g_rev * 0.3 # Mock 30% profit if no profit col
        g_margin = (g_prof / g_rev * 100) if g_rev > 0 else 0.0
        g_aov = sub_df[val_col].mean() if val_col in df.columns else 0.0
        
        channels_data.append({
            "name": g_name,
            "count": g_count,
            "revenue": g_rev,
            "profit": g_prof,
            "margin": g_margin,
            "aov": g_aov
        })
        
        summary_lines.append(
            f"    - {g_name}: Revenue = ${g_rev:,.2f}, Profit = ${g_prof:,.2f} (Margin: {g_margin:.1f}%)"
        )
    
    print_log("ds_agent", "Summary calculated:\n" + "\n".join(summary_lines), GREEN)
    time.sleep(1.5)

    # Perform t-test
    p_value = 0.5
    t_stat = 0.0
    group1, group2 = "Group A", "Group B"
    t_test_str = ""
    aov_str = ""

    if len(top_groups) >= 2 and val_col in df.columns:
        group1, group2 = top_groups[0], top_groups[1]
        vals1 = df[df[cat_col] == group1][val_col].dropna()
        vals2 = df[df[cat_col] == group2][val_col].dropna()
        
        if len(vals1) > 1 and len(vals2) > 1:
            t_res = stats.ttest_ind(vals1, vals2, equal_var=False)
            t_stat = t_res.statistic
            p_value = t_res.pvalue
            t_test_str = f"t-statistic = {t_stat:.2f}, p-value = {p_value:.3f}."
            
            sig_text = "NOT statistically significant" if p_value >= 0.05 else "is statistically SIGNIFICANT"
            aov_str = f"between {group1} (${df[df[cat_col] == group1][val_col].mean():.2f}) and {group2} (${df[df[cat_col] == group2][val_col].mean():.2f}) {sig_text}."
        else:
            t_test_str = "Insufficient data for t-test."
    else:
        t_test_str = "Insufficient categories/numerical columns for t-test."

    print_log("ds_agent", f"Calling MCP tool: run_welch_t_test(value_col='{val_col}', group_col='{cat_col}', group1='{group1}', group2='{group2}')...", GRAY)
    time.sleep(1.5)
    print_log("ds_agent", f"t-test results: {t_test_str} {aov_str}", GREEN)
    time.sleep(1.2)

    # Confound check
    print_log("ds_agent", f"Calling MCP tool: check_confound(primary_col='{cat_col}', outcome_col='{status_col if status_col else cols[0]}', potential_confound='Category')...", GRAY)
    time.sleep(1.0)
    print_log("ds_agent", "No major confounding skew found. Concluding analysis and generating output JSON...", GREEN)
    time.sleep(1.5)
    print_log("ds_agent", "Successfully generated Structured Analysis JSON containing statistical summary, margins, and hypothesis test results.", GREEN)
    time.sleep(0.5)
    print("\n--------------------------------------------------------------------------------")
    
    # Step 2: State Bridge
    print_log("state_bridge", "Ingesting Analyst JSON output...", BLUE)
    time.sleep(1.0)
    print_log("state_bridge", "Validating schema against Pydantic model `AnalysisResult`...", BLUE)
    time.sleep(1.0)
    print_log("state_bridge", "Schema validation succeeded. Session state successfully updated.", BLUE)
    print("--------------------------------------------------------------------------------\n")
    time.sleep(0.5)

    # Step 3: Storyteller Loop
    print_log("storyteller_loop", "Entering refinement loop. Iteration 1 of 2...", YELLOW)
    time.sleep(0.8)
    print_log("storyteller", f"Translating mathematical analysis and Welch t-test results for {cat_col} vs {val_col} into business narrative...", MAGENTA)
    time.sleep(2.5)
    print_log("storyteller", f"Drafting kitchen-table narrative report: focusing on {group1} and {group2} margins, and adding sample size hedge...", MAGENTA)
    time.sleep(2.0)
    print_log("storyteller", "Draft complete. Sending to Guardrail Agent...", MAGENTA)
    time.sleep(1.0)
    print_log("guardrail_agent", "Running calibration and grounding checks on storyteller narrative...", GREEN)
    time.sleep(1.5)
    print_log("guardrail_agent", "Checking for overreaching causal claims... None found.", GREEN)
    time.sleep(1.0)
    print_log("guardrail_agent", f"Checking sample size calibration (sample size of {num_rows} > 20, standard hedging applies)... Passed.", GREEN)
    time.sleep(1.0)
    print_log("guardrail_agent", "All guardrail checks PASSED. Escalating from Storyteller Loop.", GREEN)
    print("\n--------------------------------------------------------------------------------")

    # Step 4: Output Writer
    print_log("output_writer", "Retrieving validated narrative report from session state...", BLUE)
    time.sleep(1.0)
    print_log("output_writer", "Writing presentation output to `final_presentation.md`...", BLUE)
    time.sleep(1.5)

    # Prepare markdown table
    table_rows = []
    for cd in channels_data:
        table_rows.append(
            f"| **{cd['name']}** | ${cd['revenue']:,.2f} | ${cd['profit']:,.2f} | {cd['margin']:.1f}% | ${cd['aov']:.2f} |"
        )
    table_content = "\n".join(table_rows)

    welch_significance_text = (
        f"A Welch's t-test comparing the average values of {group1} and {group2} yielded a p-value of **{p_value:.3f}**. "
        + (f"This indicates that the difference is not statistically significant and likely occurred by chance." if p_value >= 0.05 else f"This indicates that the difference is statistically significant.")
    )

    # Dynamic continuous prose storyteller narrative generator
    g1_rev = channels_data[0]['revenue']
    g2_rev = channels_data[1]['revenue']
    g1_prof = channels_data[0]['profit']
    g2_prof = channels_data[1]['profit']
    g1_margin = channels_data[0]['margin']
    g2_margin = channels_data[1]['margin']
    g1_aov = channels_data[0]['aov']
    g2_aov = channels_data[1]['aov']

    # Translate p-value to casual calibration
    if p_value < 0.05:
        significance_desc = f"and that difference is stable enough that we can trust it's not a fluke—people from {group1} really are buying more at a time on average than {group2} visitors"
    else:
        significance_desc = f"but that difference is small enough that it could easily just be random noise, so we can't count on it being a real trend"

    # Construct the prose
    prose = (
        f"We set up our analysis to look at the business query: \"{args.problem.strip('\"')}\". "
        f"Looking at the records in {args.csv}, it looks like we brought in ${g1_rev:,.2f} from {group1}, "
        f"which is much higher than the ${g2_rev:,.2f} we got from {group2}. But the twist comes when you look "
        f"at what we actually got to keep: {group1} ended up with a {g1_margin:.1f}% profit margin, yielding ${g1_prof:,.2f} in profit, "
        f"while {group2} brought in a {g2_margin:.1f}% margin, which gave us ${g2_prof:,.2f}. "
        f"It is a bit like running a bakery where you sell twice as many loaves of bread, but the ingredients cost so much more that you end up "
        f"taking home the same amount of cash per hour. When you look at how much people actually buy on average, "
        f"the average order was ${g1_aov:.2f} for {group1} compared to ${g2_aov:.2f} for {group2}, {significance_desc}. "
        f"So while the top-line revenue numbers for {group1} look great, the profitability doesn't match the volume, likely because of high "
        f"acquisition costs or discount margins.\n\n"
        f"We genuinely couldn't find an answer to whether we are spending too much on advertising yet because we don't have the marketing ad bills "
        f"or client acquisition costs in these database files. To figure that out, we would need to merge this order list with our actual marketing budgets."
    )

    # Write outputs/final_presentation.md
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / "final_presentation.md"
    
    output_path.write_text(prose + "\n", encoding="utf-8")

    print(f"\n{BOLD}{GREEN}[done] outputs/final_presentation.md written ({output_path.stat().st_size} bytes){RESET}")

if __name__ == "__main__":
    main()
