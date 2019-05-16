#! /bin/bash

dir=$1

# this is old style behavior, i.e. generate symbolic links
# rm -fr $dir
# ln -s ../$dir $dir
# exit

# the new implementation follows

# remove links (possibly remaining from compilation with old makefiles):
test -L $dir && rm $dir

mkdir -p $dir

echo "This file indicates this is a temporary directory" >  $dir/.tmpdir

for file in ../$dir/*.h
do
  name="${file##*/}"
  text="#include \"../../$dir/$name\""
# files are replaces only if changed
  cmp -s <(echo "$text") $dir/$name > /dev/null  2> /dev/null || echo "$text" > $dir/$name
done

# then erase not existent files
for file in $dir/*.h
do
  name="${file##*/}"
  test -f "../$dir/$name" || rm $file
done
