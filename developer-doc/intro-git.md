\page intro-git A brief introduction to git

Clone the github repository:
\verbatim
> git clone git@github.com:plumed/plumed2.git
> cd plumed2
\endverbatim

Stay up to date:
\verbatim
> git pull
\endverbatim

Make a small fix (working locally):
\verbatim
> git add FILENAME
> git commit FILENAME
\endverbatim

Share it (needs internet connection):
\verbatim
> git pull # always check if you are up-to-date
> git push
\endverbatim

Look at what's happening:
\verbatim
> git log
\endverbatim
or better
\verbatim
> gitk --all
\endverbatim

Start working on a new feature, opening a new branch:
\verbatim
> git checkout -b new-feature
\endverbatim

List the available branches:
\verbatim
> git branch
\endverbatim

Check the available branches including the ones on the origin
\verbatim
> git branch -a
\endverbatim

Switch among branches
\verbatim
> git checkout master
> git checkout new-feature
\endverbatim

Do a commit on your new-feature branch
\verbatim
> git checkout new-feature
> # now edit NEWFILE
> git add NEWFILE
> git commit NEWFILE
\endverbatim

Merge recent work from the master branch, doing a "rebase"
\verbatim
> git checkout master
> git pull # to stay up-to-date with remote work
> git checkout new-feature
> git rebase master
\endverbatim
Notice that rebasing is only recommended if your new-feature branch
has not been shared yet with other people.

Collect the changes to the log and get rid of branches that have 
been deleted on the remote repo:
\verbatim
> git fectch --all --prune
\endverbatim

After several commits, your new feature is ready for merge.
\verbatim
# checkout the branch you want to merge your work on (e.g. master)
> git checkout master
> git pull # to stay up-to-date with remote work
> git checkout new-feature
# You can squeeze your commits with:
> git rebase -i master
# then interactively picking your commits (follow onscreen instructions)
# (Without -i, all the commits are retained)
# Merge your work into master branch
> git checkout master
> git merge new-feature
# Remove the branch
> git branch -d new-feature
# In case you want to remove a remote branch you should use:emove a remote branch
# > git push origin :crap-branch
# Analyze the results
> gitk --all
# If everything seems right, push your master branch to the central repository
> git push
\endverbatim

Checkout a remote branch and switch to it on your local repo 
\verbatim
> git checkout -b experimental origin/experimental
\endverbatim

Compare list of commits on two different branches (here branches are named master and experimental)
\verbatim
git log master..experimental
\endverbatim

All these things can be done with a GUI:
\verbatim
> git gui
\endverbatim

