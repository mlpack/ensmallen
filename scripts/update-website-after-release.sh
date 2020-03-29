#!/bin/bash
#
# This script is used to update the website after a release is made.  Push
# access to the ensmallen.org website is needed.  Generally, this script will be
# run by mlpack-bot, so it never needs to be run by hand.
#
# Usage: update-website.sh <major> <minor> <patch>

# Make sure that the mlpack repository exists.
dest_remote_name=`git remote -v |\
                  grep "https://github.com/mlpack/ensmallen (fetch)" |\
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

# Update the checked out repository, so that we can get the tags.
git fetch $dest_remote_name;

# Check out a copy of the ensmallen.org repository.
git clone https://github.com/mlpack/ensmallen.org /tmp/ensmallen.org/;

# Create the release file.
git archive --prefix=ensmallen-$MAJOR.$MINOR.$PATCH/ $MAJOR.$MINOR.$PATCH |\
    gzip > /tmp/ensmallen.org/files/ensmallen-$MAJOR.$MINOR.$PATCH.tar.gz;

# Now update the website.
wd=`pwd`;
cd /tmp/ensmallen.org/;
git add files/ensmallen-$MAJOR.$MINOR.$PATCH.tar.gz;

# Update the link to the latest version.
cd files/;
rm ensmallen-latest.tar.gz;
ln -s ensmallen-$MAJOR.$MINOR.$PATCH.tar.gz ensmallen-latest.tar.gz;
cd ../

git add files/ensmallen-latest.tar.gz;
git commit -m "Release version $MAJOR.$MINOR.$PATCH.";

# Finally, push, and we're done.
git push origin;
cd $wd;

rm -rf /tmp/ensmallen.org;
