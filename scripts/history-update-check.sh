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
    echo "HISTORY.md was updated with a change for the PR ..."
else
    echo "Please describe your PR changes in HISTORY.md ..."
    echo "Exiting CI process ... "
    exit 1
fi

exit 0
