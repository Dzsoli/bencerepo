import git

GIT_REPO_PATH = git.Repo('.', search_parent_directories=True).working_tree_dir

