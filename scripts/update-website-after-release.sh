#!/bin/bash
#
# This script is used to update the website after a release is made.  Push
# access to the ensmallen.org website is needed.  Generally, this script will be
# run by mlpack-bot, so it never needs to be run by hand.
#
# Usage: update-website-after-release.sh <major> <minor> <patch>

MAJOR=$1;
MINOR=$2;
PATCH=$3;

# Make sure that the mlpack repository exists.
dest_remote_name=`git remote -v |\
                  grep "mlpack/ensmallen (fetch)" |\
                  head -1 |\
                  awk -F' ' '{ print $1 }'`;

if [ "a$dest_remote_name" == "a" ]; then
  echo "No git remote found for mlpack/ensmallen!";
  echo "Make sure that you've got the ensmallen repository as a remote, and" \
      "that the master branch from that remote is checked out.";
  echo "You can do this with a fresh repository via \`git clone" \
      "https://github.com/mlpack/ensmallen\`.";
  exit 1;
fi

# Update the checked out repository, so that we can get the tags.
git fetch $dest_remote_name;

# Check out a copy of the ensmallen.org repository.
git clone git@github.com:mlpack/ensmallen.org /tmp/ensmallen.org/;

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

# Update the index page.
sed -i 's/\[ensmallen-[0-9]*\.[0-9]*\.[0-9]\.tar\.gz\](files\/ensmallen-[0-9]*\.[0-9]*\.[0-9]*\.tar.gz)/[ensmallen-'$MAJOR'.'$MINOR'.'$PATCH'.tar.gz](files\/ensmallen-'$MAJOR'.'$MINOR'.'$PATCH'.tar.gz)/' index.md

git add files/ensmallen-latest.tar.gz;
git add index.md;
git commit -m "Release version $MAJOR.$MINOR.$PATCH.";

# Finally, push, and we're done.
git push origin;
cd $wd;

rm -rf /tmp/ensmallen.org;
