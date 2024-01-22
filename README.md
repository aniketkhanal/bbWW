# bbWW

This package works on top of [coffea4bees](https://gitlab.cern.ch/cms-cmu/coffea4bees). 

## Instructions:

```
git clone ssh://git@gitlab.cern.ch:7999/cms-cmu/coffea4bees.git
cd coffea4bees/python/
git clone ssh://git@gitlab.cern.ch:7999/algomez/bbWW.git
```

## Set Environment

This code has been tested at the cmslpc, and to simplify the setup, it can be used with the container needed to run on lpc condor computers. To set this container under `coffea4bees/`:
```
curl -OL https://raw.githubusercontent.com/CoffeaTeam/lpcjobqueue/main/bootstrap.sh
bash bootstrap.sh
```
This creates two new files in this directory: `shell` and `.bashrc`. _Additionally, this package contains a `set_shell.sh file`_ which runs the `./shell` executable with the coffea4bees container. This container is based on the `coffeateam/coffea-dask:latest` container including some additional python packages. 
```
source set_shell.sh
```

Remember to run this previous command (aka set your environment) *every time you want to run something*.

