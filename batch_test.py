import os
import glob
from tqdm import tqdm
import pickle
import csv
import shutil


def write_to_csv(filename, content):
    file_exist = os.path.exists(filename)
    with open(filename, "a+", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=content.keys())
        if not file_exist:
            writer.writeheader()
        writer.writerow(content)


configs = glob.glob(
    "/ai/mnt/code/mmaction2/WORK_DIRS/work_dirs_update_samples_cor_binary_try*/*/*.py"
)
for config in tqdm(configs):
    print(">>>> START ON %s" % config)
    os.system("python /ai/mnt/code/mmaction2/test_bak.py --config %s" % config)
    print(">>>> DONE ON %s\n\n" % config)

configs = glob.glob(
    "/ai/mnt/code/mmaction2/WORK_DIRS/work_dirs_update_samples_cor_binary_try*/*/*.pkl"
)
savedir = (
    "/ai/mnt/code/mmaction2/WORK_DIRS/work_dirs_update_samples_cor_binary_trys_pkls"
)

os.makedirs(savedir, exist_ok=True)
for config in tqdm(configs):
    newname = config.split("/")[-1]
    if "try1" in config:
        post_fix = "_try1.pkl"
    elif "try2" in config:
        post_fix = "_try2.pkl"
    elif "try3" in config:
        post_fix = "_try3.pkl"
    else:
        assert False, "false"
    newname = newname.replace(".pkl", post_fix).split("#")[-1]
    shutil.copy(config, savedir + "/" + newname)


# methods = [
#     "c3d",
#     "i3d",
#     "r2plus1d",
#     "slowfast",
#     "swin",
#     "tanet",
#     "timesformer",
#     "tin",
#     "tsm",
#     "tsn",
#     "uniformerv2",
# ]
# savedir = os.makedirs(
#     "/ai/mnt/code/mmaction2/work_dirs_update_samples_avg", exist_ok=True
# )
# for method in tqdm(methods):
#     pkls = glob.glob(
#         "/ai/mnt/code/mmaction2/work_dirs_update_samples_try*/*/%s_*.pkl" % method
#     )
#     csv_name = savedir + "/%s.csv" % method
#     contents = []
#     logits = []
#     for pkl in pkls:
#         with open(pkl, "rb") as f:
#             content = pickle.load(f)
#             contents.append(content)
#         num_samples = len(contents[0])
#         num_results = len(pkls)

#     for s_index, s in range(num_samples):
#         logits = [0, 0, 0, 0]
#         for r_index, r in range(num_results):
#             logits[0] += r["pred_score"][0].item()
#             logits[1] += r["pred_score"][1].item()
#             logits[2] += r["pred_score"][2].item()
#             logits[3] += r["pred_score"][3].item()
#         gt = r["gt_label"].item()
#         logits = [i / num_results for i in logits]
#         pred = logits.index(max(logits))
#         newcontent = dict(
#             name=s_index,
#             T0=logits[0],
#             T2a=logits[1],
#             T2b=logits[2],
#             T3=logits[3],
#             pred=pred,
#             gt=gt,
#         )
#         write_to_csv(csv_name, content=newcontent)
