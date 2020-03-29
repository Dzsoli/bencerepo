import git
import os
import torch

GIT_REPO_PATH = git.Repo('.', search_parent_directories=True).working_tree_dir

RESULTS_PATH = os.path.abspath(os.path.join(GIT_REPO_PATH, '../results/'))
FULLDATA_PATH = os.path.abspath(os.path.join(GIT_REPO_PATH, '../full_data/'))
# print(RESULTS_PATH)

LOCAL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
