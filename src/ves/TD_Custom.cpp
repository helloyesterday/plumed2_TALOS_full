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

#include "TargetDistribution.h"
#include "GridIntegrationWeights.h"

#include "core/ActionRegister.h"
#include "tools/Grid.h"

#include "lepton/Lepton.h"


namespace PLMD {
namespace ves {

//+PLUMEDOC VES_TARGETDIST TD_CUSTOM
/*
Target distribution given by an arbitrary mathematical expression (static or dynamic).

Use as a target distribution the distribution defined by
\f[
p(\mathbf{s}) =
\frac{f(\mathbf{s})}{\int d\mathbf{s} \, f(\mathbf{s})}
\f]
where \f$f(\mathbf{s})\f$ is some arbitrary mathematical function that
is parsed by the lepton library.

The function \f$f(\mathbf{s})\f$ is given by the FUNCTION keywords by
using _s1_,_s2_,..., as variables for the arguments
\f$\mathbf{s}=(s_1,s_2,\ldots,s_d)\f$.
If one variable is not given the target distribution will be
taken as uniform in that argument.

It is also possible to include the free energy surface \f$F(\mathbf{s})\f$
in the target distribution by using the _FE_ variable. In this case the
target distribution is dynamic and needs to be updated with current
best estimate of \f$F(\mathbf{s})\f$, similarly as for the
\ref TD_WELLTEMPERED "well-tempered target distribution".
Furthermore, the inverse temperature \f$\beta = (k_{\mathrm{B}}T)^{-1}\f$ and
the thermal energy \f$k_{\mathrm{B}}T\f$ can be included
by using the _beta_ and _kBT_ variables.

The target distribution will be automatically normalized over the region on
which it is defined on. Therefore, the function given in
FUNCTION needs to be non-negative and normalizable. The
code will perform checks to make sure that this is indeed the case.


\par Examples

Here we use as shifted [Maxwell-Boltzmann distribution](https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution)
as a target distribution in one-dimension.
Note that it is not need to include the normalization factor as the distribution will be
automatically normalized.
\plumedfile
TD_CUSTOM ...
 FUNCTION=(s1+20)^2*exp(-(s1+20)^2/(2*10.0^2))
 LABEL=td
... TD_CUSTOM
\endplumedfile

Here we have a two dimensional target distribution where we
use a [generalized normal distribution](https://en.wikipedia.org/wiki/Generalized_normal_distribution)
for argument \f$s_2\f$ while the distribution for \f$s_1\f$ is taken as
uniform as the variable _s1_ is not included in the function.
\plumedfile
TD_CUSTOM ...
 FUNCTION=exp(-(abs(s2-20.0)/5.0)^4.0)
 LABEL=td
... TD_CUSTOM
\endplumedfile

By using the _FE_ variable the target distribution can depend on
the free energy surface \f$F(\mathbf{s})\f$. For example,
the following input is identical to using \ref TD_WELLTEMPERED with
BIASFACTOR=10.
\plumedfile
TD_CUSTOM ...
 FUNCTION=exp(-(beta/10.0)*FE)
 LABEL=td
... TD_CUSTOM
\endplumedfile
Here the inverse temperature is automatically obtained by using the _beta_
variable. It is also possible to use the _kBT_ variable. The following
syntax will give the exact same results as the syntax above
\plumedfile
TD_CUSTOM ...
 FUNCTION=exp(-(1.0/(kBT*10.0))*FE)}
 LABEL=td
... TD_CUSTOM
\endplumedfile


*/
//+ENDPLUMEDOC

class TD_Custom : public TargetDistribution {
private:
  void setupAdditionalGrids(const std::vector<Value*>&, const std::vector<std::string>&, const std::vector<std::string>&, const std::vector<unsigned int>&);
  //
  lepton::CompiledExpression expression;
  //
  std::vector<unsigned int> cv_var_idx_;
  std::vector<std::string> cv_var_str_;
  //
  std::string cv_var_prefix_str_;
  std::string fes_var_str_;
  std::string kbt_var_str_;
  std::string beta_var_str_;
  //
  bool use_fes_;
  bool use_kbt_;
  bool use_beta_;
public:
  static void registerKeywords( Keywords&);
  explicit TD_Custom(const ActionOptions& ao);
  void updateGrid();
  double getValue(const std::vector<double>&) const;
  ~TD_Custom() {};
};

PLUMED_REGISTER_ACTION(TD_Custom,"TD_CUSTOM")


void TD_Custom::registerKeywords(Keywords& keys) {
  TargetDistribution::registerKeywords(keys);
  keys.add("compulsory","FUNCTION","The function you wish to use for the target distribution where you should use the variables _s1_,_s2_,... for the arguments. You can also use the current estimate of the FES by using the variable _FE_ and the temperature by using the _kBT_ and _beta_ variables.");
  keys.use("WELLTEMPERED_FACTOR");
  keys.use("SHIFT_TO_ZERO");
}


TD_Custom::TD_Custom(const ActionOptions& ao):
  PLUMED_VES_TARGETDISTRIBUTION_INIT(ao),
//
  cv_var_idx_(0),
  cv_var_str_(0),
//
  cv_var_prefix_str_("s"),
  fes_var_str_("FE"),
  kbt_var_str_("kBT"),
  beta_var_str_("beta"),
//
  use_fes_(false),
  use_kbt_(false),
  use_beta_(false)
{
  std::string func_str;
  parse("FUNCTION",func_str);
  checkRead();
  //
  try {
    lepton::ParsedExpression pe=lepton::Parser::parse(func_str).optimize(lepton::Constants());
    log<<"  function as parsed by lepton: "<<pe<<"\n";
    expression=pe.createCompiledExpression();
  }
  catch(PLMD::lepton::Exception& exc) {
    plumed_merror("There was some problem in parsing the function "+func_str+" given in FUNCTION with lepton");
  }

  for(auto &p: expression.getVariables()) {
    std::string curr_var = p;
    unsigned int cv_idx;
    if(curr_var.substr(0,cv_var_prefix_str_.size())==cv_var_prefix_str_ && Tools::convert(curr_var.substr(cv_var_prefix_str_.size()),cv_idx) && cv_idx>0) {
      cv_var_idx_.push_back(cv_idx-1);
    }
    else if(curr_var==fes_var_str_) {
      use_fes_=true;
      setDynamic();
      setFesGridNeeded();
    }
    else if(curr_var==kbt_var_str_) {
      use_kbt_=true;
    }
    else if(curr_var==beta_var_str_) {
      use_beta_=true;
    }
    else {
      plumed_merror(getName()+": problem with parsing formula with lepton, cannot recognise the variable "+curr_var);
    }
  }
  //
  std::sort(cv_var_idx_.begin(),cv_var_idx_.end());
  cv_var_str_.resize(cv_var_idx_.size());
  for(unsigned int j=0; j<cv_var_idx_.size(); j++) {
    std::string str1; Tools::convert(cv_var_idx_[j]+1,str1);
    cv_var_str_[j] = cv_var_prefix_str_+str1;
  }
}


void TD_Custom::setupAdditionalGrids(const std::vector<Value*>& arguments, const std::vector<std::string>& min, const std::vector<std::string>& max, const std::vector<unsigned int>& nbins) {
  if(cv_var_idx_.size()>0 && cv_var_idx_[cv_var_idx_.size()-1]>getDimension()) {
    plumed_merror(getName()+": mismatch between CVs given in FUNC and the dimension of the target distribution");
  }
}


double TD_Custom::getValue(const std::vector<double>& argument) const {
  plumed_merror("getValue not implemented for TD_Custom");
  return 0.0;
}


void TD_Custom::updateGrid() {
  if(use_fes_) {
    plumed_massert(getFesGridPntr()!=NULL,"the FES grid has to be linked to the free energy in the target distribution");
  }
  if(use_kbt_) {
    try {
      expression.getVariableReference(kbt_var_str_) = 1.0/getBeta();
    } catch(PLMD::lepton::Exception& exc) {}
  }
  if(use_beta_) {
    try {
      expression.getVariableReference(beta_var_str_) = getBeta();
    } catch(PLMD::lepton::Exception& exc) {}
  }
  //
  std::vector<double> integration_weights = GridIntegrationWeights::getIntegrationWeights(getTargetDistGridPntr());
  double norm = 0.0;
  //
  for(Grid::index_t l=0; l<targetDistGrid().getSize(); l++) {
    std::vector<double> point = targetDistGrid().getPoint(l);
    for(unsigned int k=0; k<cv_var_str_.size() ; k++) {
      try {
        expression.getVariableReference(cv_var_str_[k]) = point[cv_var_idx_[k]];
      } catch(PLMD::lepton::Exception& exc) {}
    }
    if(use_fes_) {
      try {
        expression.getVariableReference(fes_var_str_) = getFesGridPntr()->getValue(l);
      } catch(PLMD::lepton::Exception& exc) {}
    }
    double value = expression.evaluate();

    if(value<0.0 && !isTargetDistGridShiftedToZero()) {plumed_merror(getName()+": The target distribution function gives negative values. You should change the definition of the function used for the target distribution to avoid this. You can also use the SHIFT_TO_ZERO keyword to avoid this problem.");}
    targetDistGrid().setValue(l,value);
    norm += integration_weights[l]*value;
    logTargetDistGrid().setValue(l,-std::log(value));
  }
  if(norm>0.0) {
    targetDistGrid().scaleAllValuesAndDerivatives(1.0/norm);
  }
  else if(!isTargetDistGridShiftedToZero()) {
    plumed_merror(getName()+": The target distribution function cannot be normalized proberly. You should change the definition of the function used for the target distribution to avoid this. You can also use the SHIFT_TO_ZERO keyword to avoid this problem.");
  }
  logTargetDistGrid().setMinToZero();
  // Added by Y. Isaac Yang to calculate the reweighting factor
  if(isReweightGridActive())
  {
	std::vector<double> rw_integration_weights = GridIntegrationWeights::getIntegrationWeights(getReweightGridPntr());
    double rw_norm = 0.0;
  //
    for(Grid::index_t l=0; l<reweightGrid().getSize(); l++) {
      std::vector<double> rw_point = reweightGrid().getPoint(l);
      for(unsigned int k=0; k<cv_var_str_.size() ; k++) {
        try {
          expression.getVariableReference(cv_var_str_[k]) = rw_point[cv_var_idx_[k]];
        }   catch(PLMD::lepton::Exception& exc) {}
      }
      if(use_fes_) {
        try {
          expression.getVariableReference(fes_var_str_) = getFesRWGridPntr()->getValue(l);
        } catch(PLMD::lepton::Exception& exc) {}
      }
      double rw_value = expression.evaluate();

      if(rw_value<0.0 && !isTargetDistGridShiftedToZero()) {plumed_merror(getName()+": The reweight target distribution function gives negative values. You should change the definition of the function used for the target distribution to avoid this. You can also use the SHIFT_TO_ZERO keyword to avoid this problem.");}
      reweightGrid().setValue(l,rw_value);
      rw_norm += rw_integration_weights[l]*rw_value;
      logReweightGrid().setValue(l,-std::log(rw_value));
    }
    if(rw_norm>0.0) {
      reweightGrid().scaleAllValuesAndDerivatives(1.0/rw_norm);
    }
    else if(!isTargetDistGridShiftedToZero()) {
      plumed_merror(getName()+": The target distribution function cannot be normalized proberly. You should change the definition of the function used for the target distribution to avoid this. You can also use the SHIFT_TO_ZERO keyword to avoid this problem.");
    }
    logReweightGrid().setMinToZero();
  }
  //
}


}
}
