
if [ $# -eq 0 ]; then
    exit 1
fi
if [ "$CONDA_DEFAULT_ENV" = "$1" ]; then
    conda deactivate
fi
builtin cd "$(dirname "${BASH_SOURCE:-$0}")"
ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT" || exit 1

conda remove --name $1 --all -y
rm -rf src/
rm -rf srtml_exp.egg-info/