#!/usr/bin/env python3

# SCRIPT pour rattraper le fait que les last_time_step_multiplied_loss ne sont pas calculées. Calcul et ajoute juste cette métrique au fichier de best score (pas results table)
# à lancer au même niveau dans l'arborescence qu'un fichier "run_results_table" c'et à dire dans un répertoire de run.
import numpy as np

def main():
    """"""

    lts_mul_loss = {}
    lts_mul_loss_list = []
    with open("run_results_table.csv", 'r') as f_table:
        lines = f_table.readlines()

        headers = lines[0].split(',')

        col_lts_mse = headers.index("eval_regression_output_last_time_step_mse")
        col_lts_cc = headers.index("eval_event_output_last_time_step_CategoricalCrossentropy")

        for i in range(1, len(lines)):
            splitted = lines[i].split(',')
            lts_mul_loss[f"fold_{i}"] = np.float32(splitted[col_lts_mse])*np.float32(splitted[col_lts_cc])
            lts_mul_loss_list.append(lts_mul_loss[f"fold_{i}"])

        lts_mul_loss_list = sorted(lts_mul_loss_list)

    with open("best_run_models.txt", 'a') as f_best:
        f_best.write("eval_last_time_step_multiplied_loss\n")
        f_best.write("ranking,name,value\n")
        
        for i, values in enumerate(lts_mul_loss_list):
            for k, v in lts_mul_loss.items():
                if values == v:
                    f_best.write(f"{i},{k},{v}\n")

if __name__ == "__main__":
    main()