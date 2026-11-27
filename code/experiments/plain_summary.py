"""Summary of the plain loading runs. Usage, from the code folder: python -m experiments.plain_summary

Published fills are from Alvarez Martinez, Alvarez-Valdes and Parreno (2015), Pesquisa Operacional
35(1), read from the article page. They are means over the 100 instances of a class, for one
customer, with orientation, full support AND load-bearing constraints, which this run does not have:
  Table 6: their GRASP and the GRASP of Alonso et al. (2014), classes 1 to 15;
  Table 2, row for one customer: their GRASP on the instances of Christensen and Rousoe (2009), classes 1 to 7.

Fills for the same problem as this run, with orientation and the support constraint and nothing
else, are from Table 4 of Goncalves and Resende (2010), AT&T Labs Research technical report,
optimization-online.org/2010/07/2688, read from the PDF on 27 Nov 2026: their BRKGA-CLP-S, and the
packing variant of the tree search of Fanslau and Bortfeldt (2010) as they report it. Means over
the 100 instances of a class.
"""
import pandas as pd

from .common import read

BRKGA_S = [94.34, 94.88, 95.05, 94.75, 94.58, 94.39, 93.74, 92.65, 91.90, 91.28, 90.39, 89.81, 89.27, 88.57, 87.96]
TREE_SEARCH = [94.51, 94.73, 94.74, 94.41, 94.13, 93.85, 93.20, 92.26, 91.48, 90.86, 90.11, 89.51, 88.98, 88.26, 87.57]
GRASP_2015 = [82.5, 86.7, 88.4, 88.1, 87.4, 86.9, 85.53, 83.7, 82.4, 81.4, 80.1, 78.9, 77.7, 76.6, 75.5]
ALONSO_2014 = [81.4, 85.7, 87.3, 86.9, 86.6, 86.3, 85.7, 85.5, 84.8, 84.0, 82.8, 81.3, 80.2, 78.9, 78.0]
GRASP_2015_ONE_CUSTOMER = [92.47, 92.75, 92.76, 92.22, 91.68, 91.01, 89.62]


def main(name="plain.csv"):
    d = read(name)
    budgets = sorted(d["per_type"].unique())
    print(f"{d.groupby(['br_class', 'instance']).ngroups} runs: {d['br_class'].nunique()} classes, "
          f"{d.groupby('br_class')['instance'].nunique().iloc[0]} instances each; budgets per box type {[int(b) for b in budgets]}\n")
    fill = d.pivot_table(index=["br_class", "types"], columns="per_type", values="fill") * 100
    print("Fill (%), mean over the instances of a class, by evaluations per box type:")
    print(fill.round(2).to_string())
    print("\nGain of each doubling:")
    print(fill.diff(axis=1).iloc[:, 1:].round(2).to_string())

    last = d[d["per_type"] == budgets[-1]]
    by_class = last.groupby("br_class")
    classes = sorted(last["br_class"].unique())
    table = pd.DataFrame({
        "types": by_class["types"].first(),
        "fill": by_class["fill"].mean() * 100,
        "lowest": by_class["fill"].min() * 100,
        "highest": by_class["fill"].max() * 100,
        "standard error": by_class["fill"].sem() * 100,
        "boxes loaded (%)": (by_class["loaded"].sum() / by_class["boxes"].sum()) * 100,
        "seconds": by_class["seconds"].mean(),
        "GRASP 2015": pd.Series(GRASP_2015, index=range(1, 16)).reindex(classes),
        "Alonso et al. 2014": pd.Series(ALONSO_2014, index=range(1, 16)).reindex(classes),
        "GRASP 2015, C&R instances": pd.Series(GRASP_2015_ONE_CUSTOMER, index=range(1, 8)).reindex(classes),
    })
    table["minus GRASP 2015"] = table["fill"] - table["GRASP 2015"]
    table["minus Alonso et al."] = table["fill"] - table["Alonso et al. 2014"]
    print(f"\nAt {budgets[-1]} evaluations per box type, against published fills (see the head of this file):")
    print(table.round(2).to_string())
    for label, group in (("classes 1 to 7", [c for c in classes if c <= 7]), ("classes 8 to 15", [c for c in classes if c >= 8]),
                         ("all classes", classes)):
        if group:
            part = table.loc[group]
            print(f"{label}: fill {part['fill'].mean():.2f}; GRASP 2015 {part['GRASP 2015'].mean():.2f}; "
                  f"Alonso et al. 2014 {part['Alonso et al. 2014'].mean():.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="plain.csv")
    main(parser.parse_args().file)
