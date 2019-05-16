/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2016-2017 The VES code team
   (see the PEOPLE-VES file at the root of this folder for a list of names)

   See http://www.ves-code.org for more information.

   This file is part of VES code module.

   The VES code module is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   The VES code module is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with the VES code module.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

#include "VesBias.h"
#include "LinearBasisSetExpansion.h"
#include "CoeffsVector.h"
#include "CoeffsMatrix.h"
#include "BasisFunctions.h"
#include "Optimizer.h"
#include "TargetDistribution.h"
#include "VesTools.h"

#include "bias/Bias.h"
#include "core/ActionRegister.h"
#include "core/ActionSet.h"
#include "core/PlumedMain.h"


namespace PLMD {
namespace ves {

class VesLinearExpansion : public VesBias {
private:
  unsigned int nargs_;
  // the value of coefficient of each basis function (added by YI Yang)
  std::vector<double> coeffsderivs_values_store;
  std::vector<BasisFunctions*> basisf_pntrs_;
  LinearBasisSetExpansion* bias_expansion_pntr_;
  size_t ncoeffs_;
  Value* valueForce2_;
public:
  explicit VesLinearExpansion(const ActionOptions&);
  ~VesLinearExpansion();
  void calculate();
  void updateTargetDistributions();
  void restartTargetDistributions();
  //
  void setupBiasFileOutput();
  void writeBiasToFile();
  void resetBiasFileOutput();
  //
  void setupFesFileOutput();
  void writeFesToFile();
  void resetFesFileOutput();
  //
  void setupFesProjFileOutput();
  void writeFesProjToFile();
  //
  void writeTargetDistToFile();
  void writeTargetDistProjToFile();
  //
  double calculateReweightFactor() const;
  //
  static void registerKeywords( Keywords& keys );
  std::vector<BasisFunctions*> get_basisf_pntrs() const {return basisf_pntrs_;}
  LinearBasisSetExpansion* get_bias_expansion_pntr() const {return bias_expansion_pntr_;}
  // the value of coefficient of each basis function (added by YI Yang)
  void getBasisSetValues(std::vector<double>& bs_values) {bs_values=coeffsderivs_values_store;}
  // Added by Y. Isaac Yang to calculate the reweighting factor
  void updateReweightingFactor();
  //
};

}
}
