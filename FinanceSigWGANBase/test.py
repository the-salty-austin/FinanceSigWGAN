# import glob
# import os
#
# from lib.datasets import DATA_DIR
#
#
# datasets = os.listdir(DATA_DIR)
#
# for dataset in datasets:
#     datadir = os.path.join(DATA_DIR, dataset)
#     files = glob.glob(os.path.join(datadir, '*csv'))
#     files = files + glob.glob(os.path.join(datadir, '*', '*csv'))
#     files = [f.replace('datasets', '').replace('\\', '.') for f in files]
#     print(dataset)
#     for f in files:
#         print(f'\t{f}')
#     print()

a = list(range(10))
b = list(range(5, 8))
c = list(set(a) - set(b))
print(c)