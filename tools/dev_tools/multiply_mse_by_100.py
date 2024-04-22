#!/usr/bin/env python3

# multiply by 100 every metric and losses using mse in a run_results_table.csv file

import numpy as np

def main():
    """"""

    # read file and get headers and values
    headers = []
    values = []
    with open("run_results_table.csv", 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.rsplit('\n')[0].split(',')
            
            if i == 0:
                headers = line
                continue

            #else
            values.append(line)
    
    # multiply mse losses and metrics and evaluation multiplied losses by 100
    for i in range(0, len(values)):
        for j in range(0, len(values[i])):
            if headers[j] in ["regression_output_loss", "val_regression_output_loss", "eval_regression_output_loss", "regression_output_last_time_step_mse", "val_regression_output_last_time_step_mse", "eval_regression_output_last_time_step_mse", "eval_multiplied_loss", "eval_last_time_step_multiplied_loss"]:
                values[i][j] = np.float32(values[i][j])*100

    # re write file
    with open("run_results_table2.csv", "w") as f2:
        for k, head in enumerate(headers):
            if k == len(headers)-1:
                f2.write(f"{head}\n")
                break    
            f2.write(f"{head},")

        for i in range(0, len(values)):
            for j in range(0, len(values[i])):
                if j == len(values[i]) - 1:
                    f2.write(f"{values[i][j]}\n")
                    break
                f2.write(f"{values[i][j]},")

    print("[END] Successfully created run_results_tables2.csv\n")

if __name__ == "__main__":
    main()