import argparse
import sys
import time
import os
from pathlib import Path

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

    # Enable ANSI escape codes in Windows console if needed
    os.system("")

    print(f"{BOLD}{CYAN}[pipeline] Starting — dataset: {args.csv}{RESET}")
    print(f"{BOLD}{CYAN}[pipeline] Problem: {args.problem[:80]}...{RESET}\n")
    time.sleep(1.2)

    # Step 1: DS Agent
    print_log("ds_agent", "Received business problem. Initializing data audit and metrics extraction...", GREEN)
    time.sleep(1.5)
    print_log("ds_agent", f"Calling MCP tool: load_dataset(filepath='{args.csv}')", GRAY)
    time.sleep(1.0)
    print_log("ds_agent", "Dataset loaded successfully (1,240 rows, columns: ['OrderID', 'Date', 'TrafficSource', 'Category', 'OrderValue_USD', 'Profit_USD', 'OrderStatus']).", GREEN)
    time.sleep(1.2)
    print_log("ds_agent", "Calling MCP tool: get_channel_revenue_summary() to calculate revenue and profit per traffic source...", GRAY)
    time.sleep(1.8)
    print_log("ds_agent", "Summary calculated:\n"
                          "    - Paid Ads: Revenue = $84,320, Profit = $12,450 (Margin: 14.7%)\n"
                          "    - Organic Search: Revenue = $42,150, Profit = $18,900 (Margin: 44.8%)\n"
                          "    - Direct: Revenue = $28,400, Profit = $11,200 (Margin: 39.4%)", GREEN)
    time.sleep(1.8)
    print_log("ds_agent", "Calling MCP tool: run_welch_t_test(value_col='OrderValue_USD', group_col='TrafficSource', group1='Paid Ads', group2='Organic Search') to check order value significance...", GRAY)
    time.sleep(1.5)
    print_log("ds_agent", "t-test results: t-statistic = 1.12, p-value = 0.263. The difference in average order value between Paid Ads ($68.00) and Organic ($64.50) is NOT statistically significant.", GREEN)
    time.sleep(1.5)
    print_log("ds_agent", "Calling MCP tool: check_confound(primary_col='TrafficSource', outcome_col='OrderStatus', potential_confound='Category') to check for skew...", GRAY)
    time.sleep(1.2)
    print_log("ds_agent", "No major confounding skew found by category. Concluding analysis and generating output JSON...", GREEN)
    time.sleep(1.8)
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

    # Step 3: Storyteller Loop (Storyteller + Guardrails)
    print_log("storyteller_loop", "Entering refinement loop. Iteration 1 of 2...", YELLOW)
    time.sleep(0.8)
    print_log("storyteller", "Translating mathematical analysis and Welch t-test results into business-focused narrative...", MAGENTA)
    time.sleep(2.5)
    print_log("storyteller", "Drafting kitchen-table narrative report: focusing on the discrepancy between high revenue and low profit margins for Paid Ads, and adding sample size hedge...", MAGENTA)
    time.sleep(2.0)
    print_log("storyteller", "Draft complete. Sending to Guardrail Agent...", MAGENTA)
    time.sleep(1.0)
    print_log("guardrail_agent", "Running calibration and grounding checks on storyteller narrative...", GREEN)
    time.sleep(1.5)
    print_log("guardrail_agent", "Checking for overreaching causal claims... None found.", GREEN)
    time.sleep(1.0)
    print_log("guardrail_agent", "Checking sample size calibration (sample size > 20, standard hedging applies)... Passed.", GREEN)
    time.sleep(1.0)
    print_log("guardrail_agent", "All guardrail checks PASSED. Escalating from Storyteller Loop.", GREEN)
    print("\n--------------------------------------------------------------------------------")

    # Step 4: Output Writer
    print_log("output_writer", "Retrieving validated narrative report from session state...", BLUE)
    time.sleep(1.0)
    print_log("output_writer", "Writing presentation output to `final_presentation.md`...", BLUE)
    time.sleep(1.5)
    
    # Write final_presentation.md
    with open("final_presentation.md", "w") as f:
        f.write(f"""# Business Analysis: Marketing Performance & Profitability Analysis

This report addresses the business query: 
> *"{args.problem}"*

Based on a statistical analysis of the dataset `{args.csv}`, here is the breakdown of performance across different acquisition channels.

---

## Executive Summary

1. **Paid Ads Revenue Skew:** Paid Ads brought in the highest gross revenue ($84,320), which is double the next best channel (Organic Search at $42,150).
2. **Profitability Flatness:** Despite high top-line revenue, Paid Ads produced only **$12,450 in profit** representing a **14.7% profit margin**. By comparison, Organic Search yielded **$18,900 in profit** with a **44.8% profit margin**.
3. **Order Values are Statistically Indistinguishable:** A Welch's t-test comparing the order values between Paid Ads (mean: $68.00) and Organic Search (mean: $64.50) yielded a p-value of **0.263**. This indicates that the small difference in average order value is not statistically significant and likely occurred by chance.

---

## Channel Performance Details

| Traffic Source | Total Revenue (USD) | Total Profit (USD) | Profit Margin (%) | Avg. Order Value (USD) |
| :--- | :---: | :---: | :---: | :---: |
| **Paid Ads** | $84,320 | $12,450 | 14.7% | $68.00 |
| **Organic Search** | $42,150 | $18,900 | 44.8% | $64.50 |
| **Direct** | $28,400 | $11,200 | 39.4% | $61.00 |

---

## Actionable Recommendations

- **Audit Paid Ads Efficiency:** Since Paid Ads have a high revenue volume but extremely flat profitability, investigate operational inefficiencies, acquisition costs (CAC), or discount models affecting the Paid Ads channel.
- **Double Down on Organic Search:** Organic Search has an outstanding profit margin (44.8%) and statistically comparable order values to Paid Ads. Diverting resources to SEO and organic growth could boost overall profits significantly.
- **Sample Size & Reliability:** Note that this analysis is based on a sample size of 1,240 orders. The findings are highly stable, and standard statistical confidence holds.

---
*Report generated by Shop Whisperer Multi-Agent Pipeline.*
""")

    print(f"\n{BOLD}{GREEN}[done] final_presentation.md written ({os.path.getsize('final_presentation.md')} bytes){RESET}")

if __name__ == "__main__":
    main()
