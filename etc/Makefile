# Command aliases
CONDA_EXE ?= conda
CONDA_LOCK := $(CONDA_EXE)-lock

name = $(word 1, $(subst ., ,$1))

ENVIRONMENT_YMLS = $(wildcard *.environment.yml)
ENVIRONMENTS := $(foreach e,$(wildcard *.environment.yml),$(call name, $e))
LOCKFILES := $(foreach e, $(ENVIRONMENTS), $e.conda-lock.yml)
# Explicit lockfiles are used in github actions with conda-incubator/setupminiconda
EXPLICIT_LOCKFILES := $(foreach e, $(ENVIRONMENTS), $e.linux-64.lock)

help:  ## Display help on all Makefile targets
	@@grep -h '^[a-zA-Z]' $(MAKEFILE_LIST) | awk -F ':.*?## ' 'NF==2 {printf "   %-20s%s\n", $$1, $$2}' | sort

.PHONY: $(ENVIRONMENTS)
$(ENVIRONMENTS):  ## Lock this specific environment.yml file
	rm -f $@.conda-lock.yml
	rm -f $@.linux-64.lock
	$(CONDA_LOCK) lock -f $@.environment.yml --lockfile $@.conda-lock.yml
	$(CONDA_LOCK) render -p linux-64 --filename-template "$@.{platform}.lock" $@.conda-lock.yml

lock: $(ENVIRONMENTS)  ## lock all .environment.yml files

clean:  ## Remove lockfiles and explicit files
	rm $(LOCKFILES)
	rm $(EXPLICIT_LOCKFILES)
