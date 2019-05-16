\page AddingAColvar How to add a new collective variable

To implement a CV you need to create a single cpp file called <i>ColvarName</i>.cpp in the directory src/colvar. 
In addition, if you would like us to incorporate your CV in the release version of PLUMED you will need to write at least
one regression test for your CV.  Details on how to write regression tests are provided here: \ref regtests

If you use the following template for your new <i>ColvarName</i>.cpp file then the manual and the calls to the CV will 
be looked after automatically.

\verbatim
#include "Colvar.h"
#include "ActionRegister.h"

#include <string>
#include <cmath>
#include <cassert>

using namespace std;

namespace PLMD{
namespace colvar{

//+PLUMEDOC COLVAR NAME
/*
\endverbatim

At this point you provide the description of your CV that will appear in the manual along with an description of the input file syntax and an example.  Merging new features of the code into the plumed main branch without proper documentation is punishable by death!  Some instructions as to how to format this information is provided here: \ref usingDoxygen

\verbatim
*/
//+ENDPLUMEDOC

/**** We begin by declaring a class for your colvar.  This class inherits everything from the Colvar class.
      This ensures it has a label, a place to store its value, places to the store the values of the derivatives
      and that it can access the various atoms it will employ.

class ColvarNAME : public Colvar {
\endverbatim

Insert declarations for your colvar's parameters here using plumed's \ref parsing.

\verbatim
public:
 /---- This routine is used to create the descriptions of all the keywords used by your CV
 static void registerKeywords( Keywords& keys ); 
 /---- This is the constructor for your colvar.  It is this routine that will do all the reading.
       Hence it takes as input a line from the input file.
  ColvarNAME(const ActionOptions&);
 /---- This is the routine that will be used to calculate the value of the colvar, whenever its calculation is required.
       This routine and the constructor above must be present - if either of them are not the code will not compile.
  virtual void calculate();
};

 /------ The following command inserts your new colvar into plumed by inserting calls to your new
        routines into the parts of plumed where they are required.  This macro takes two arguments:
        The first is the name of your ColvarClass and the second is the keyword for your CV
        (the first word in the input line for your CV).
PLUMED_REGISTER_ACTION(ColvarNAME,"KEYWORD")

 /----- The following routine creates the documentation for the keyowrds used by your CV
void ColvarName::registerKeywords( Keywords& keys ){
  Colvar::registerKeywords(keys);
\endverbatim

In here you should add all your descriptions of the keywords used by your colvar as well as descriptions of any components
that you can use this colvar to calculate. Descriptions as to how to do this can be found here: \ref usingDoxygen

\verbatim
}

 /---- We now write the actual readin (constructor) and calculations routines for the colvar

ColvarName::ColvarName(const ActionOptions&ao):
 /------ This line sets up various things in the plumed core which colvars rely on.
PLUMED_COLVAR_INIT(ao)
{
 vector<int> atoms;  /----- You almost always have atoms -----/
\endverbatim

Insert code here to read the arguments of the CV here using plumed's parsing functionality.  N.B.  The label is read in already elsewhere.

\verbatim
  checkRead();     /--- This command checks that everything on the input line has been read properly

/--- The following two lines inform the plumed core that we require space to store the value
     of the CV and that the CV will act on a particular list of atoms.
  addValueWithDerivatives("");
  requestAtoms(atoms);

/ --- For a number of the free energy methods in plumed it is necessary to calculate the
      distance between two points in CV space.  Obviously, for periodic CVs one must take
      periodicities into account when calculating distances and use the minimum image
      convention in distance calculations.  Hence, we set the periodicity of the cv using
      the following two lines.
   getValue("")->setPeridodicity(true);  // Set this true if the CV is periodic otherwise set if false.
   getValue("")->setDomain(min,max);     // This routine is only required if the function is periodic.  It sets the minimum and maximum values of the colvar.
}

void ColvarName::calculate(){
/--- These are the things you must calculate for any cv ---/
  double cv_val;              /--- The value of the cv ----/
  Tensor boxDerivatives;      /--- The derivative of the cv with respect to the box vectors ----/
  vector<double> derivatives; /--- The derivative of the cv with respect to the atom positions ---/
\endverbatim

Insert the code to calculate your cv, its derivatives and its contribution to the virial here. Please use, where possible, the library of tools described in \ref TOOLBOX.

\verbatim
/---- Having calculated the cv, its derivative and the contribution to the virial you now
      transfer this information to the plumed core using the following three commands. 
  for(int i=0;i<derivatives.size();i++){ setAtomsDerivatives(i,derivatives[i]); }
  setBoxDerivatives(boxDerivatives);
  setValue(cv_val);
}
\endverbatim

\section multicvs Mult-component CVs

To avoid code duplication, and in some cases computational expense, plumed has functionality so that a single line in input can calculate be used to calculate multiple components for a CV.  For example, PATH computes the distance along the path,\f$s\f$, and the distance from the path, \f$z\f$.  Alternatively, a distance can give one the \f$x\f$, \f$y\f$ and \f$z\f$ components of the vector connecting the two atoms.  You can make use of this functionality in your own CVs as follows:

- In the constructor we create an additional value for the CV by adding the call PLMD::addValueWithDerivative("new") as well as PLMD::addValueWithDerivatives(””).  In addtion set any periodicity for our component using getValue("new")->setPeridicity() and getValue("new")->setDomain(min,max).  If this CV is called plum in our input file we can now use both plum and plum.new in any of the functions/methods in plumed.
- Obviously in calculate we now must provide functionality to calculate the values, boxDerivatives and the atom derivatives for both our original plum and its component plum.new. Furthermore, all of this data must be transferred to the plumed core.  This is done by the following code:

Here we transfer the value, box derivatives and atomic derivatives for plum.
\verbatim
for(int i=0;i<derivatives.size();i++){ setAtomsDerivatives(i,derivatives[i]); }
setBoxDerivatives(boxDerivatives);
setValue(cv_val);
\endverbatim
Here we transfer the value, box derivatives and atomic derivatives for plum.new.
\verbatim
Value* nvalue=getValue("new");
for(int i=0;i<nderivatives.size();i++){ setAtomsDerivatives(nvalue i,nderivatives[i]); }
setBoxDerivatives(nvalue,nboxDerivatives);
setValue(nvalue,ncv_val);
\endverbatim

Please only use this functionality for CVs that are VERY similar.
