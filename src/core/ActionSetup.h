/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2011-2019 The plumed team
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
#ifndef __PLUMED_core_ActionSetup_h
#define __PLUMED_core_ActionSetup_h

#include "Action.h"

namespace PLMD {

/**
\ingroup MULTIINHERIT
Action used to create a PLMD::Action that do something during setup only e.g. PLMD::SetupUnits
*/
class ActionSetup :
  public virtual Action {
public:
/// Constructor
  explicit ActionSetup(const ActionOptions&ao);
/// Creator of keywords
  static void registerKeywords( Keywords& keys );
/// Do nothing.
  void calculate() {}
/// Do nothing.
  void apply() {}
};

}

#endif
