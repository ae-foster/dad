import pickle
import datetime
import subprocess


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def get_git_diff():
    return subprocess.check_output(['git', 'diff'])


def persist_output_to_filename(output, experiment_name):
    output_dir = "./run_outputs/dad/"
    if not experiment_name:
        experiment_name = output_dir + "{}".format(datetime.datetime.now().isoformat())
    else:
        experiment_name = output_dir + experiment_name
    results_file = experiment_name + '.pickle'
    output['git-hash'] = get_git_revision_hash()
    output['git-diff'] = get_git_diff()

    with open(results_file, 'wb') as f:
        pickle.dump(output, f)

