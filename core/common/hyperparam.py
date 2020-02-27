import git
import os

GIT_REPO_PATH = git.Repo('.', search_parent_directories=True).working_tree_dir

RESULTS_PATH = os.path.abspath(os.path.join(GIT_REPO_PATH, '../results/'))

