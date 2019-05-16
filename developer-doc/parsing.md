\page parsing Parsing functionality

By now you are probably familiar with the way that plumed2 input looks:

\verbatim
DISTANCE ATOMS=0,300 LABEL=dist NOPBC
RESTRAINT ARG=dist KAPPA=1.0 AT=1.0
\endverbatim

Within the code this information will be read by either parseVector or parseFlag.  parseFlag is called using:

\verbatim
parseFlag("NOPBC",nopbc)
\endverbatim

This will then read the list of action objects you passed to the constructor and look for the keyword NOPBC.  If the keyword is found then it is deleted from the list of action objects, while the boolian nopbc is returned as true otherwise the boolian is returned as false.  parseVector is called using:

\verbatim
std::vector<double> vec;
parseVector("KAPPA",vec);
\endverbatim

This routine will then read the list of action objects you passed to the constructor and look for the keyword KAPPA.  This keyword will be followed by an equals sign and a list of comma separated numbers.  These numbers are read into the vector vec and passed back to the main code.  (N.B.  The size of the vector is worked out automitically by parseVector from the input.  In addition the vector can be a vector of int or a vector of real.)  Much like parseFlag, parseVector will delete the keyword and the list from the list of actionObjects once it has completed.  When you have finished reading all your arguments you should call checkRead() - this routine checks that everything in the list of ActionOptions taken from the input has been read in. 

Please note when you are implementing functionality to read the plumed input that you never need to implement anything to read ARGS and LABEL as these keywords are read elsewhere in the code. 

