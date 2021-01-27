from mnultitool.matrix import frobeniusFromPolyCoeffs

datasets = [[1, 2, 3, 4, 5], [2, 2, 3, 4, 5, 6]]

for dataset in datasets:
    print(dataset, ":\n", frobeniusFromPolyCoeffs(dataset), "\n")
