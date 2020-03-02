#!/usr/bin/env bash
#
# Check each PR has an entry in HISTORY.md during a CI routine.
#
# Arguments:
#   $ history-update-check.sh
#
# This should be run from the root of the repository.
res=$(git diff origin/master --name-only | grep ^HISTORY.md | wc -l)
echo "Files Changed:"
git diff --name-only
if [ $res -gt 0 ]; then
    echo "HISTORY.md was updated with a change for the PR ..."
else
    echo "Please describe your PR changes in HISTORY.md ..."
    echo "Exiting CI process ... "
    exit 1
fi

exit 0
