check_command_exist() {
    
    case "$1" in
        go)
            if ! [ -x "$(command -v $1)" ]; then
                wget -q -O - https://raw.githubusercontent.com/canha/golang-tools-install-script/master/goinstall.sh | bash
            fi 
            ;;
        docker)
            if ! [ -x "$(command -v $1)" ]; then
                wget -q -O -  https://get.docker.com/ |  bash
            fi 
            ;;
        *)
            echo "$1 is not a required dependency"
            exit 1
    esac
        
}
if [ $# -eq 0 ]; then
    exit 1
fi
check_command_exist go
check_command_exist docker
builtin cd "$(dirname "${BASH_SOURCE:-$0}")"
ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT" || exit 1

conda create --name $1 python=3.6 -y
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate $1
pip install -e git+https://github.com/gatech-sysml/srtml.git#egg=srtml[torch,tensorflow,cuda101,dev] --verbose
pip install -r "$ROOT/requirements.txt"
pip install -e . --verbose
if [ $# -eq 2 ]; then
    if [ $2 == "localstack_init" ]; then
        wget -O - https://gist.githubusercontent.com/alindkhare/37e23c9a364552bcaebe469752e2de17/raw/10c907d5ce43ad6b2cb5e219e93103fdcef67c60/start_localstack.sh | bash
    fi
fi

