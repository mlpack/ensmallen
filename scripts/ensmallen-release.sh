#!/usr/bin/env bash
#
# Release a new version of ensmallen.
#
# Arguments:
#   $ ensmallen-release.sh <github username> <major> <minor> <patch> [<name>]
#
# This should be run from the root of the repository.
set -e

if [ "$#" -lt 4 ]; then
  echo "At least four arguments required!";
  echo "$ ensmallen-release.sh <github username> <major> <minor> <patch>" \
      "[<name>]";
  exit 1;
fi

if [ "$#" -gt 5 ]; then
  echo "Too many arguments!";
  echo "$ ensmallen-release.sh <github username> <major> <minor> <patch>" \
      "[<name>]";
  exit 1;
fi

# Make sure that the branch is clean.
# Truncate leading whitespaces since wc -l on MacOS adds an extra \t.
lines=`git diff | wc -l | sed -e 's/^\s*//g'`;
if [ "$lines" != "0" ]; then
  echo "git diff returned a nonzero result!";
  echo "";
  git diff;
  exit 1;
fi

# Make sure that the mlpack repository exists.
dest_remote_name=`git remote -v |\
                  grep "mlpack/ensmallen (fetch)" |\
                  head -1 |\
                  awk -F' ' '{ print $1 }'`;

if [ "a$dest_remote_name" == "a" ]; then
  echo "No git remote found for https://github.com/mlpack/ensmallen!";
  echo "Make sure that you've got the ensmallen repository as a remote, and" \
      "that the master branch from that remote is checked out.";
  echo "You can do this with a fresh repository via \`git clone" \
      "https://github.com/mlpack/ensmallen\`.";
  exit 1;
fi

# Also check that we're on the master branch, from the correct origin.
current_branch=`git branch --no-color | grep '^\* ' | awk -F' ' '{ print $2 }'`;
current_origin=`git rev-parse --abbrev-ref --symbolic-full-name @{u} |\
                awk -F'/' '{ print $1 }'`;
if [ "a$current_branch" != "amaster" ]; then
  echo "Current branch is $current_branch."
  echo "This script has to be run from the master branch."
  exit 1;
elif [ "a$current_origin" != "a$dest_remote_name" ]; then
  echo "Current branch does not track from remote mlpack repository!";
  echo "Instead, it tracks from $current_origin/master.";
  echo "Make sure to check out a branch that tracks $dest_remote_name/master.";
  exit 1;
fi

# Make sure 'gh' is installed.
hub_output="`which hub`" || true;
if [ "a$hub_output" == "a" ]; then
  echo "The Hub command-line tool must be installed for this script to run" \
      "successfully.";
  echo "See https://hub.github.com for more details and installation" \
      "instructions.";
  echo "";
  echo "(apt-get install hub on Debian and Ubuntu)";
  echo "(brew install hub via Homebrew)";
  exit 1;
fi

# Check git remotes: we need to make sure we have a fork to push to.
github_user=$1;
remote_name=`git remote -v |\
             grep "$github_user/ensmallen (push)" |\
             head -1 |\
             awk -F' ' '{ print $1 }'`;
if [ "a$remote_name" == "a" ]; then
  echo "No git remote found for $github_user/ensmallen!";
  echo "Adding remote '$github_user'.";
  git remote add $github_user https://github.com/$github_user/ensmallen;
  remote_name="$github_user";
fi
git fetch $github_user;

# Make sure everything is up to date.
git pull;

# Make updates to files that will be needed for the release.
MAJOR=$2;
MINOR=$3;
PATCH=$4;

sed -i 's/ENS_VERSION_MAJOR[ ]*[0-9]*$/ENS_VERSION_MAJOR '$MAJOR'/' \
    include/ensmallen_bits/ens_version.hpp;
sed -i 's/ENS_VERSION_MINOR[ ]*[0-9]*$/ENS_VERSION_MINOR '$MINOR'/' \
    include/ensmallen_bits/ens_version.hpp;
sed -i 's/ENS_VERSION_PATCH[ ]*[0-9]*$/ENS_VERSION_PATCH '$PATCH'/' \
    include/ensmallen_bits/ens_version.hpp;

if [ "$#" -eq "5" ]; then
  sed -i 's/ENS_VERSION_NAME[ ]*\".*\"$/ENS_VERSION_NAME \"'"$5"'\"/' \
      include/ensmallen_bits/ens_version.hpp;
fi

# Update CONTRIBUTING.md.
sed -i "s/ensmallen-[0-9]*\.[0-9]*\.[0-9]*/ensmallen-$MAJOR.$MINOR.$PATCH/g" \
    CONTRIBUTING.md;

# Update HISTORY.md with the release date and possibly name.
version_name=`grep ENS_VERSION_NAME include/ensmallen_bits/ens_version.hpp |\
              head -1 |\
              sed 's/.*\"\(.*\)\"/\1/'`;
year=`date +%Y`;
month=`date +%m`;
day=`date +%d`;
new_line="ensmallen $MAJOR.$MINOR.$PATCH: \"$version_name\"";
sed -i "s/### ensmallen ?.??.?: \"???\"/### $new_line/" HISTORY.md;
sed -i "s/###### ????-??-??/###### $year-$month-$day/" HISTORY.md;

# Now, we'll do all this on a new release branch.
git checkout -b release-$MAJOR.$MINOR.$PATCH;

git add include/ensmallen_bits/ens_version.hpp;
git add CONTRIBUTING.md;
git add HISTORY.md;
git commit -m "Update and release version $MAJOR.$MINOR.$PATCH.";

changelog_str=`cat HISTORY.md |\
    awk '/^### /{f=0} /^### ensmallen '"$MAJOR"'.'"$MINOR"'.'"$PATCH"': "'"$version_name"'"/{f=1} f{print}' |\
    grep -v '^#' |\
    tr '\n' '!' |\
    sed -e 's/!  [ ]*/ /g' |\
    tr '!' '\n'`;
echo "Changelog string:"
echo "$changelog_str"

# Add one more commit to create the new HISTORY block.
echo "### ensmallen ?.??.?: \"???\"" > HISTORY.md.new;
echo "###### ????-??-??" >> HISTORY.md.new;
echo "" >> HISTORY.md.new;
cat HISTORY.md >> HISTORY.md.new;
mv HISTORY.md.new HISTORY.md;
git add HISTORY.md;
git commit -m "Add new block for next release to HISTORY.md.";

# Push to new branch.
git push --set-upstream $github_user release-$MAJOR.$MINOR.$PATCH;

# Next, we have to actually open the PR for the release.  These lines would be
# hard to wrap so they are longer than the length limit. :)
hub pull-request \
    -b mlpack:master \
    -h $github_user:release-$MAJOR.$MINOR.$PATCH \
    -m "Release version $MAJOR.$MINOR.$PATCH: \"$version_name\"" \
    -m "This automatically-generated pull request adds the commits necessary to make the $MAJOR.$MINOR.$PATCH release." \
    -m "Once the PR is merged, mlpack-bot will tag the release as HEAD~1 (so that it doesn't include the new HISTORY block) and publish it." \
    -m "Or, well, hopefully that will happen someday." \
    -m "When you merge this PR, be sure to merge it using a *rebase*." \
    -m "### Changelog" \
    -m "$changelog_str" \
    -l "t: release"

echo "";
echo "Switching back to 'master' branch.";
echo "If you want to access the release branch again, use \`git checkout" \
    "release-$MAJOR.$MINOR.$PATCH\`.";
exit 0;
