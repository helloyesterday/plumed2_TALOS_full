\page AddingACLTool How to add a new command-line tool

To implement a command line tool you need to create a single cpp file call CLToolNAME.cpp. You can, in a command line
tool, use functionality from plumed to perform simple post-processing tasks.  For example, sum_hills uses the 
functionality inside from inside the biasing PLMD::Action, PLMD::BiasMetaD to calculate free energy surfaces.
Regardless, of what you are endeavoring to do your CLToolNAME.cpp file should be formatted in accordance with
the following template:

\verbatim
#include "CLTool.h"
#include "CLToolRegister.h"
#include "PlumedConfig.h"
#include "ActionRegister.h"

using namespace std;

namespace PLMD {
/**
//+PLUMEDOC TOOLS name
\endverbatim

Insert the documentation for your new tool here

\verbatim
\par Examples
\endverbatim

Insert some examples of how to use your tool here 

\verbatim
*/
//+ENDPLUMEDOC

/******* This is how you should declare the class for your command line tool.  main() does
         all the analsysis you require. The constructor and the registerKeywords routine 
         only do anything if you are using one of the template forms of input described below.
         However, reigsterKeywords() must always be declared.

class CLToolNAME:
public CLTool
{
public:
  static void registerKeywords( Keywords& keys );
  CLToolNAME(const CLToolOptions& co );
  int main(FILE* in, FILE*out,Communicator& pc);
  string description()const{
    return "a description of the particular kind of task CLToolNAME performs";
  }
};

PLUMED_REGISTER_CLTOOL(CLToolNAME,"name")
\endverbatim

Insert the code for main, registerKeywords and the constructor here.

\verbatim
}  /--- don't forget to close the namespace PLMD { at the end of the file
\endverbatim

\section input Input

The input stream is passed to main so you can create tools that have an input in any form.
However, there are certain standardized forms of input that can be used for command line
tools. If your tool takes its input in one of these forms we strongly encourage you to
use the code that is already present.  If you do so a great deal of manual generation will 
be looked after automatically.

There are two command line input types that are implemented in the base class.  The first
is for tools such as driver where the input is specified using a series of command line flags
e.g.

\verbatim
plumed driver --plumed plumed.dat --ixyz trajectory.xyz --dump-forces
\endverbatim

The other are tools like simplmd that take an input file that contains one directive per line
and a corresponding value for that directive.  For both these forms of input it is possible to
read in everything that is required to run the calculation prior to the actual running of the 
calculation.  In other words these the user is not prompted to provide input data once the main 
calculation has started running (N.B. you can do tools with input of this sort though as the input stream
is passed to main).  

If you wish to use driver-like or simple-md like input then you have to specify this in the constructor.
For driver-like input you would write the following for the constructor:

\verbatim
CLToolNAME::CLToolNAME(const CLToolOptions& co ):
CLTool(co)
{
 inputdata=commandline;
}
\endverbatim

For simplemd-like input you write the following for the constructor:

\verbatim
CLToolNAME( const CLToolOptions& co ) :
CLTool(co)
{
  inputdata=ifile;
}
\endverbatim

If you are not using one of these input forms then you don't need to write a constructor although
you may choose to for reasons not connected to the input of data.

Once you have created the constructor the actual readin and manual creation is done in much the same
manner as it is done for Actions in the main plumed code (\ref usingDoxygen). You write a
registerKeywords routine as follows:

\verbatim
void CLToolNAME::registerKeywords( Keywords& keys ){
  CLTool::registerKeywords( keys );
}  
\endverbatim

Inside this routine you add descriptions of all your various command-line flags (driver-like) or
input directives (simple-md-like) and these descriptions are used to create the manual. The code 
will automatically check if a particular flag is present and read any input directive connected 
with a particular flag (i.e. the data after the space). Within main you can recover the read in 
data using CLTool::parse and CLTool:parseFlag.   

\section getplumed Re-using plumed

To re-use the functionality that is present in plumed you use the same tools that are used to 
patch the various MD codes (\ref HowToPlumedYourMD).  Alternatively, if you want to create
an instance of a particular Action you can do so by issuing the following commands:

\verbatim
PlumedMain* plumed=new PlumedMain(); std::vector<std::string> words;
Action* action=actionRegister().create(ActionOptions(plumed,words));
delete plumed; delete action;
\endverbatim

Please be aware that words should contain everything that would be required in an input
line to make a valid instance of the Action you require.
