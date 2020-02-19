#!/usr/bin/env bash
#
# Check each PR has an entry in HISTORY.md during a CI routine.
#
# Arguments:
#   $ history-update-check.sh
#
# This should be run from the root of the repository.
set -e

! git diff --exit-code master -- HISTORY.md > /dev/null

if [ $? ]; then
    echo "History log was updated ..."
else
    echo "History log is missing an entry for this PR ..."
    echo "Please update the history log to run CI ..."
    exit 1
fi

exit 0
