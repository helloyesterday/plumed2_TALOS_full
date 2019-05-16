/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2015-2019 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#ifndef __PLUMED_pamm_HBPammObject_h
#define __PLUMED_pamm_HBPammObject_h

#include "tools/Vector.h"
#include "multicolvar/AtomValuePack.h"
#include "PammObject.h"

namespace PLMD {
namespace pamm {

class HBPammObject {
private:
/// The Pamm object underlying this HBPamm calculation
  PammObject mypamm;
/// Pointer to base class in multicolvar
  multicolvar::MultiColvarBase* mymulti;
public:
/// Setup the HBPamm object
  void setup( const std::string& filename, const double& reg, multicolvar::MultiColvarBase* mybase, std::string& errorstr );
/// Get the cutoff to use throughout
  double get_cutoff() const ;
/// Evaluate the HBPamm Object
  double evaluate( const unsigned& dno, const unsigned& ano, const unsigned& hno,
                   const Vector& d_da, const double& md_da, multicolvar::AtomValuePack& myatoms ) const ;
};

}
}

#endif
