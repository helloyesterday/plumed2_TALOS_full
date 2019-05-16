\page AddingAFunction How to add a new function

Many collective variables are a function of a number of some set of simpler collective variables. 
These sorts of collective variables should be implemented should be implemented in plumed as functions so as not to duplicate code.

Much like CVs you can implement a function by creating a single cpp file called FunctionNAME.cpp.  
If you would like us to incorporate your CV in the release version of PLUMED you will need to write at least
one regression test for your function.  Details on how to write regression tests are provided here: \ref regtests

If you use the following template for your new FunctionNAME.cpp file then the manual and the calls to the CV will be looked after automatically.

\verbatim
#include "ActionRegister.h"
#include "Function.h"

#include <cmath>
#include <cassert>
using namespace std;
namespace PLMD{

//+PLUMEDOC FUNCTION COMBINE
/*
\endverbatim

At this point you provide the description of your function that will appear in the manual along with a description of the input file syntax and an example.  Merging new features of the code into the plumed main branch without proper documentation is punishable by death!  Some instructions as to how to format this information is provided here: \ref usingDoxygen

\verbatim
*/
//+ENDPLUMEDOC

/**** We begin by declaring a class for your function.  This class inherits everything from the function class.
      This ensures it has a label, a place to store its value, places to the store the values of the derivatives
      and that it knows which of the colvar objects its value depends on.
class FunctionNAME :
  public Function
{
\endverbatim

Insert declarations for your function's parameters here.

\verbatim
  /---- This routine is used to create the descriptions of all the keywords used by your new function 
  static void registerKeywords( Keywords& keys );
  /---- This is the constructor for your function.  It is this routine that will do all the reading.
       Hence it takes as input a line from the input file.
  FunctionNAME(const ActionOptions&);
  /---- This is the routine that will be used to calculate the value of the function, whenever its calculation is required.
        This routine and the constructor above must be present - if either of them are not the code will not compile.
  void calculate();
};

  /------ The following command inserts your new function into plumed by inserting calls to your new
          routines into the parts of plumed where they are required.  This macro takes two arguments:
          The first is the name of your FunctionClass and the second is the keyword for your function
          (the first word in the input line for your function).
PLUMED_REGISTER_ACTION(FunctionNAME,"COMBINE")

/----- The following routine creates the documentation for the keyowrds used by your CV
void FunctionNAME::registerKeywords( Keywords& keys ){
  Function::registerKeywords(keys);
\endverbatim

In here you should add all your descriptions of the keywords used by your colvar. Descriptions as to how to
do this can be found here: \ref usingDoxygen

\verbatim
}

FunctionNAME::FunctionNAME(const ActionOptions&ao):
/--- These two lines set up various things in the plumed core whcih functions rely on.
Action(ao),
Function(ao)
{
\endverbatim

Insert code here to read the arguments of the function here using plumed's \ref parsing.  N.B. The label and arguments (i.e. the cvs on which the function depends are read in already elsewhere.

\verbatim

/---- For a number of the free energy methods in plumed it is necessary to calculate the
      distance between two points in CV space.  Obviously, for periodic CVs one must take
      periodicities into account when calculating distances and use the minimum image
      convention in distance calculations.  Functions too are used as cvs in these methods
      and thus it is necessary to provide periodicities for these objects too.  In theory it
      should be possible to determine the periodicity of a function from the periodicity of the
      underlying CVs.  However, in practise this is very difficult to do.  We therefore recommend
      that you include the following few lines of code so that the periodicity of functions can
      be specified by the user in input.
  vector<string> period;
  double min(0),max(0);
  parseVector("PERIODIC",period);
  if(period.size()==0){
  }else if(period.size()==1 && period[0]=="NO"){
    getValue("")->setPeriodicity(false);
  } else if(period.size()==2 && Tools::convert(period[0],min) && Tools::convert(period[1],max)){
    getValue("")->setPeriodicity(true);
    getValue("")->setDomain(min,max);
  }
  checkRead();    /--- This command checks that everything on the input line has been read properly

  /--- The following line informs the plumed core that we require space to store the
       value of the function and the derivatives. 
  addValueWithDerivatives("");
}

\verbatim
void FunctionCombine::calculate(){
/--- These are the things you must calculate for any function ---/
  double cv_val;              /--- The value of the function ----/
  vector<double> derivatives; /--- The derivative of the function with respect to the cvs ---/
\endverbatim

Insert code here to calculate your function and its derivatives with repsect to the underlying cvs here. Please use, where possible, the library of tools described in \ref TOOLBOX.

\verbatim
  /---- Having calculated the function and its derivatives you now transfer this information
        to the plumed core using the following two commands. 
  for(int i=0;i<derivatives.size();i++){ setAtomsDerivatives(i,derivatives[i]); }
  setValue(cv_val);
}

}
\endverbatim

\section multicvsf Multi-component functions

To avoid code duplication, and in some cases computational expense, plumed has functionality so that a single line in input can calculate be used to calculate multiple components for a function.  You can make use of this functionality in your own CVs as follows:

- In the constructor we create an additional value for the function by adding the call PLMD::addValueWithDerivative("new") as well as PLMD::addValueWithDerivatives(””).  Please note you must provide keywords in input to provide periodicity information for both your function and all of its components.  The periodicities of the components should then be set using getValue("new")->setPeridicity() and getValue("new")->setDomain(min,max). If we call this function flum in our input file we can now use both flum and flum.new in any of the functions/methods in plumed.
- Obviously in calculate we now must provide functionality to calculate the values and derivatives for both flum and its component flum.new. Furthermore, all of this data must be transferred to the plumed core.  This is done by the following code:

Here we transfer the value and derivatives for flum.
\verbatim
for(int i=0;i<derivatives.size();i++){ setDerivatives(i,derivatives[i]); }
setValue(cv_val);
\endverbatim
Here we transfer the value and derivatives for plum.new.
\verbatim
Value* nvalue=getValue("new");
for(int i=0;i<nderivatives.size();i++){ setDerivatives(nvalue i,nderivatives[i]); }
setValue(nvalue,ncv_val);
\endverbatim

Please only use this functionality for functions that are VERY similar.

